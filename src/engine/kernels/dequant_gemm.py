"""Fused AWQ int4 dequantize + GEMM in Triton.

The weight matrix is never materialized in higher precision in global memory:
each program loads a tile of packed int32 words, unpacks the 4-bit values,
applies zero-point and scale in registers, and feeds the tile straight to
``tl.dot``. For batch-1 decode (M = 1) the kernel is memory bound, so the number
of bytes read from DRAM (~0.5 B per weight plus scales/zeros) is what matters.

AWQ "GEMM" checkpoint layout (verified against AutoAWQ 0.2.9, ``WQLinear_GEMM``)::

    qweight : int32 [K, N // 8]       8 x 4-bit values per word
    qzeros  : int32 [K // G, N // 8]  same packing
    scales  : fp16  [K // G, N]
    dequant : w[k, n] = (q[k, n] - z[k // G, n]) * s[k // G, n]

Nibble ``i`` (bits ``4i..4i+3``) of word ``c`` holds output column
``8*c + AWQ_ORDER[i]`` with ``AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)``. Equivalently,
output column ``n`` lives in nibble ``AWQ_REVERSE_ORDER[n % 8]`` of word ``n // 8``.

This module also holds the pure-torch packing / reference dequantization used by
the tests and by the reference execution mode of the model.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

AWQ_PACK = 8
AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
AWQ_REVERSE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


# --------------------------------------------------------------------------- #
# Pure-torch packing / reference                                              #
# --------------------------------------------------------------------------- #
def _pack_columns(vals: torch.Tensor) -> torch.Tensor:
    """Pack ``[R, N]`` integers in [0, 15] into ``[R, N // 8]`` int32 (AWQ order)."""
    rows, n = vals.shape
    if n % AWQ_PACK:
        raise ValueError(f"N={n} must be a multiple of {AWQ_PACK}")
    v = vals.to(torch.int64).reshape(rows, n // AWQ_PACK, AWQ_PACK)
    out = torch.zeros(rows, n // AWQ_PACK, dtype=torch.int64, device=vals.device)
    for i, col in enumerate(AWQ_ORDER):
        out |= v[:, :, col] << (4 * i)
    out = torch.where(out >= 2**31, out - 2**32, out)  # two's-complement wrap
    return out.to(torch.int32)


def pack_awq(int_weight: torch.Tensor, zeros: torch.Tensor, group_size: int):
    """Pack integer weights ``[K, N]`` and zero points ``[K // G, N]`` (values 0..15)."""
    k, n = int_weight.shape
    if k % group_size or zeros.shape != (k // group_size, n):
        raise ValueError("zeros must have shape [K // group_size, N]")
    if int_weight.min() < 0 or int_weight.max() > 15 or zeros.min() < 0 or zeros.max() > 15:
        raise ValueError("4-bit values must lie in [0, 15]")
    return _pack_columns(int_weight), _pack_columns(zeros)


def unpack_awq(qweight: torch.Tensor, qzeros: torch.Tensor, group_size: int):
    """Return integer weights ``[K, N]`` and zeros ``[K // G, N]`` (int32, values 0..15)."""
    shifts = torch.tensor(AWQ_REVERSE_ORDER, device=qweight.device, dtype=torch.int32) * 4

    def _unpack(packed: torch.Tensor) -> torch.Tensor:
        rows, words = packed.shape
        expanded = packed.unsqueeze(-1).expand(rows, words, AWQ_PACK)
        return ((expanded >> shifts) & 0xF).reshape(rows, words * AWQ_PACK)

    return _unpack(qweight), _unpack(qzeros)


def dequantize_reference(qweight, qzeros, scales, group_size: int) -> torch.Tensor:
    """fp16 weights ``[K, N]``: ``(q - z) * s`` with the product rounded to fp16.

    This mirrors AutoAWQ's ``dequantize_gemm`` (integer subtraction, then a single
    fp16 multiply by the fp16 scale).
    """
    iw, iz = unpack_awq(qweight, qzeros, group_size)
    iz = iz.repeat_interleave(group_size, dim=0)
    s = scales.repeat_interleave(group_size, dim=0)
    return (iw - iz).to(torch.float16) * s


def awq_gemm_reference(x, qweight, qzeros, scales, bias=None, group_size: int = 128):
    """Reference ``x @ dequant(W) (+ bias)`` in fp32 on the fp16 dequantized weights."""
    w = dequantize_reference(qweight, qzeros, scales, group_size)
    y = x.reshape(-1, x.shape[-1]).float() @ w.float()
    if bias is not None:
        y = y + bias.float()
    return y


# --------------------------------------------------------------------------- #
# Triton kernels                                                              #
# --------------------------------------------------------------------------- #
@triton.jit
def _awq_gemm_kernel(
    x_ptr,
    qw_ptr,
    qz_ptr,
    s_ptr,
    bias_ptr,
    y_ptr,
    part_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_ym,
    HAS_BIAS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)  # packed columns
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_c = offs_c < N // 8

    # Output column n sits in nibble REVERSE_ORDER[n % 8] = (j % 2) * 4 + j // 2.
    j = tl.arange(0, BLOCK_N) % 8
    shifts = ((j % 2) * 4 + j // 2) * 4

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-blocks are dealt round-robin to the SPLIT_K programs along grid axis 2.
    n_blocks = K // BLOCK_K
    n_iters = (n_blocks - pid_k + SPLIT_K - 1) // SPLIT_K
    for it in range(0, n_iters):
        k0 = (pid_k + it * SPLIT_K) * BLOCK_K
        offs_k = k0 + tl.arange(0, BLOCK_K)

        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )

        # One load per packed word, then each word is repeated 8x along the
        # columns and the right nibble is shifted out per column.
        qw = tl.load(
            qw_ptr + offs_k[:, None] * (N // 8) + offs_c[None, :],
            mask=mask_c[None, :],
            other=0,
        )
        qw = tl.interleave(qw, qw)
        qw = tl.interleave(qw, qw)
        qw = tl.interleave(qw, qw)
        w_int = (qw >> shifts[None, :]) & 0xF

        # BLOCK_K divides GROUP_SIZE, so the whole tile shares one group row.
        offs_g = k0 // GROUP_SIZE + tl.arange(0, 1)
        qz = tl.load(
            qz_ptr + offs_g[:, None] * (N // 8) + offs_c[None, :],
            mask=mask_c[None, :],
            other=0,
        )
        qz = tl.interleave(qz, qz)
        qz = tl.interleave(qz, qz)
        qz = tl.interleave(qz, qz)
        z_int = (qz >> shifts[None, :]) & 0xF

        s = tl.load(
            s_ptr + offs_g[:, None] * N + offs_n[None, :],
            mask=mask_n[None, :],
            other=0.0,
        )

        w = (w_int - z_int).to(tl.float16) * s  # fp16, exactly as the reference
        acc = tl.dot(x, w, acc)

    mask_out = mask_m[:, None] & mask_n[None, :]
    if SPLIT_K == 1:
        if HAS_BIAS:
            b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            acc += b[None, :]
        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :],
            acc.to(y_ptr.dtype.element_ty),
            mask=mask_out,
        )
    else:
        tl.store(
            part_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=mask_out,
        )


@triton.jit
def _splitk_reduce_kernel(
    part_ptr,
    bias_ptr,
    y_ptr,
    MN,
    N,
    HAS_BIAS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(0, SPLIT_K):
        acc += tl.load(part_ptr + s * MN + offs, mask=mask, other=0.0)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs % N, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs, acc.to(y_ptr.dtype.element_ty), mask=mask)


# --------------------------------------------------------------------------- #
# Launcher                                                                    #
# --------------------------------------------------------------------------- #
def _num_sms(device: torch.device) -> int:
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).multi_processor_count
    return 24  # CPU / interpreter: value only shapes the grid, not the result


CONFIG_KEYS = {"BLOCK_M", "BLOCK_N", "BLOCK_K", "SPLIT_K", "num_warps", "num_stages"}
REQUIRED_CONFIG_KEYS = {"BLOCK_M", "BLOCK_N", "BLOCK_K", "SPLIT_K"}


def validate_config(cfg: dict, k: int, group_size: int) -> dict:
    """Check a launch configuration (e.g. one read from a tuned-config file) and return a copy.

    ``num_warps`` / ``num_stages`` are optional Triton launch options; leaving them out keeps
    Triton's defaults. Raises ``ValueError`` on unknown keys or values that would mis-launch.
    """
    unknown = set(cfg) - CONFIG_KEYS
    missing = REQUIRED_CONFIG_KEYS - set(cfg)
    if unknown or missing:
        raise ValueError(f"bad awq_gemm config {cfg}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    for key, value in cfg.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"bad awq_gemm config {cfg}: {key} must be a positive int")
    if cfg["BLOCK_N"] % 8 or cfg["BLOCK_K"] % 16:
        raise ValueError(f"bad awq_gemm config {cfg}: BLOCK_N must be a multiple of 8 and BLOCK_K of 16")
    if group_size % cfg["BLOCK_K"] or k % cfg["BLOCK_K"]:
        raise ValueError("BLOCK_K must divide both group_size and K")
    if cfg["SPLIT_K"] > k // cfg["BLOCK_K"]:
        raise ValueError(f"SPLIT_K {cfg['SPLIT_K']} exceeds the {k // cfg['BLOCK_K']} K-blocks of this layer")
    return dict(cfg)


def default_config(m: int, n: int, k: int, group_size: int, device: torch.device) -> dict:
    """Untuned heuristic. ``scripts/tune_kernels.py`` measures better ones on the target GPU."""
    block_k = min(group_size, 128)
    block_m = 16 if m <= 16 else 64
    block_n = 64
    tiles = triton.cdiv(n, block_n) * triton.cdiv(m, block_m)
    n_blocks = k // block_k
    split_k = 1
    # Aim for ~4 programs per SM so enough bytes are in flight to cover DRAM latency.
    while split_k < 16 and tiles * split_k < 4 * _num_sms(device) and n_blocks // (split_k * 2) >= 1:
        split_k *= 2
    return {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k, "SPLIT_K": split_k}


def awq_gemm(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    group_size: int = 128,
    out_dtype: torch.dtype | None = None,
    config: dict | None = None,
) -> torch.Tensor:
    """``y = x @ dequant(W) + bias`` without materializing ``dequant(W)``.

    ``x`` is fp16 ``[..., K]``; the result has shape ``[..., N]`` and dtype
    ``out_dtype`` (default: ``x.dtype``). Accumulation is fp32.
    """
    if x.dtype != torch.float16:
        raise TypeError("awq_gemm expects fp16 activations")
    lead = x.shape[:-1]
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    n = qweight.shape[1] * AWQ_PACK
    if qweight.shape[0] != k:
        raise ValueError(f"qweight rows {qweight.shape[0]} != K {k}")
    if qzeros.shape != (k // group_size, n // AWQ_PACK) or scales.shape != (k // group_size, n):
        raise ValueError("qzeros / scales shape does not match qweight and group_size")
    if k % group_size or group_size % 16:
        raise ValueError("K must be a multiple of group_size and group_size a multiple of 16")

    cfg = validate_config(config, k, group_size) if config else default_config(m, n, k, group_size, x.device)
    if group_size % cfg["BLOCK_K"] or k % cfg["BLOCK_K"]:
        raise ValueError("BLOCK_K must divide both group_size and K")
    split_k = cfg["SPLIT_K"]
    launch = {key: cfg[key] for key in ("num_warps", "num_stages") if key in cfg}

    y = torch.empty((m, n), dtype=out_dtype or x.dtype, device=x.device)
    part = (
        torch.empty((split_k, m, n), dtype=torch.float32, device=x.device)
        if split_k > 1
        else y  # unused placeholder pointer when SPLIT_K == 1
    )
    bias_arg = bias if bias is not None else y

    grid = (triton.cdiv(n, cfg["BLOCK_N"]), triton.cdiv(m, cfg["BLOCK_M"]), split_k)
    _awq_gemm_kernel[grid](
        x2, qweight, qzeros, scales, bias_arg, y, part,
        m, n, k, x2.stride(0), y.stride(0),
        HAS_BIAS=bias is not None,
        GROUP_SIZE=group_size,
        SPLIT_K=split_k,
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        BLOCK_K=cfg["BLOCK_K"],
        **launch,
    )
    if split_k > 1:
        block = 1024
        _splitk_reduce_kernel[(triton.cdiv(m * n, block),)](
            part, bias_arg, y, m * n, n,
            HAS_BIAS=bias is not None, SPLIT_K=split_k, BLOCK=block,
        )
    return y.reshape(*lead, n)
