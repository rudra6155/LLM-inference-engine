"""Fused decode attention (FlashAttention-style online softmax) in Triton.

Scope: batch size 1, one query token (decode), grouped-query attention. One program
handles one KV head (and one slice of the sequence, see below) and treats the
``G = H_q / H_kv`` query heads that share it as the rows of a (padded) tile, so K and
V are read from DRAM once per KV head rather than once per query head. K and V are
streamed in ``BLOCK_N``-token blocks with a running max / running sum, so the
``[G, seq_len]`` score matrix never exists in global memory.

Two options exist for running inside a CUDA graph and for using more of the GPU:

* ``seq_len`` may be a one-element ``int32`` **device tensor** instead of a Python int.
  A CUDA graph freezes kernel arguments, so a Python ``seq_len`` would be replayed
  forever; reading it from memory lets the same graph serve every position.
* ``num_splits > 1`` is split-KV ("flash decoding"): the sequence is cut into
  ``num_splits`` contiguous slices handled by separate programs, whose partial
  results (un-normalized accumulator, running max, running sum) are merged by a
  second small kernel. With ``H_kv = 2`` the unsplit kernel launches only two
  programs, far fewer than the GPU has SMs. ``num_splits = 1`` (the default) is the
  original single-kernel path, unchanged.

``num_warps`` / ``num_stages`` are passed to Triton as launch options; ``None`` keeps
Triton's defaults.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

MIN_DOT_ROWS = 16  # tl.dot requires M >= 16; query-head groups are padded to this


@triton.jit(do_not_specialize=["seq_len"])
def _decode_attn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    pacc_ptr,
    pml_ptr,
    seqlen_ptr,
    seq_len,
    sm_scale,
    stride_qh,
    stride_kh,
    stride_ks,
    stride_vh,
    stride_vs,
    stride_oh,
    H_Q,
    GROUP: tl.constexpr,
    G_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SEQ_FROM_PTR: tl.constexpr,
):
    kv_head = tl.program_id(0)
    split = tl.program_id(1)

    if SEQ_FROM_PTR:
        seq_len = tl.load(seqlen_ptr)

    # This program's slice [lo, hi) of the sequence. With NUM_SPLITS == 1 it is [0, seq_len).
    chunk = tl.cdiv(tl.cdiv(seq_len, NUM_SPLITS), BLOCK_N) * BLOCK_N
    lo = split * chunk
    hi = tl.minimum(lo + chunk, seq_len)

    offs_g = tl.arange(0, G_PAD)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_g = offs_g < GROUP
    q_head = kv_head * GROUP + offs_g

    q = tl.load(
        q_ptr + q_head[:, None] * stride_qh + offs_d[None, :],
        mask=mask_g[:, None],
        other=0.0,
    )  # [G_PAD, HEAD_DIM]

    m_i = tl.full((G_PAD,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((G_PAD,), dtype=tl.float32)
    acc = tl.zeros((G_PAD, HEAD_DIM), dtype=tl.float32)

    k_base = k_ptr + kv_head * stride_kh
    v_base = v_ptr + kv_head * stride_vh

    for start in range(lo, hi, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < hi

        # K block loaded transposed: [HEAD_DIM, BLOCK_N]
        k = tl.load(
            k_base + offs_n[None, :] * stride_ks + offs_d[:, None],
            mask=mask_n[None, :],
            other=0.0,
        )
        s = tl.dot(q, k) * sm_scale  # [G_PAD, BLOCK_N], fp32
        s = tl.where(mask_n[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(
            v_base + offs_n[:, None] * stride_vs + offs_d[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )  # [BLOCK_N, HEAD_DIM]
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    if NUM_SPLITS == 1:
        out = acc / l_i[:, None]
        tl.store(
            o_ptr + q_head[:, None] * stride_oh + offs_d[None, :],
            out.to(o_ptr.dtype.element_ty),
            mask=mask_g[:, None],
        )
    else:
        # Un-normalized partial result of this slice; an empty slice stores m = -inf, l = 0, acc = 0,
        # which the merge kernel weights by exp(-inf) = 0.
        tl.store(
            pacc_ptr + (split * H_Q + q_head[:, None]) * HEAD_DIM + offs_d[None, :],
            acc,
            mask=mask_g[:, None],
        )
        tl.store(pml_ptr + (split * H_Q + q_head) * 2, m_i, mask=mask_g)
        tl.store(pml_ptr + (split * H_Q + q_head) * 2 + 1, l_i, mask=mask_g)


@triton.jit
def _decode_attn_merge_kernel(
    pacc_ptr,
    pml_ptr,
    o_ptr,
    stride_oh,
    H_Q,
    HEAD_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NS_PAD: tl.constexpr,
):
    h = tl.program_id(0)
    offs_s = tl.arange(0, NS_PAD)
    mask_s = offs_s < NUM_SPLITS
    m = tl.load(pml_ptr + (offs_s * H_Q + h) * 2, mask=mask_s, other=float("-inf"))
    l = tl.load(pml_ptr + (offs_s * H_Q + h) * 2 + 1, mask=mask_s, other=0.0)
    m_max = tl.max(m, axis=0)
    w = tl.exp(m - m_max)  # an empty slice has m = -inf, so its weight is 0
    denom = tl.sum(l * w, axis=0)
    offs_d = tl.arange(0, HEAD_DIM)
    acc = tl.load(
        pacc_ptr + (offs_s[:, None] * H_Q + h) * HEAD_DIM + offs_d[None, :],
        mask=mask_s[:, None],
        other=0.0,
    )
    out = tl.sum(acc * w[:, None], axis=0) / denom
    tl.store(o_ptr + h * stride_oh + offs_d, out.to(o_ptr.dtype.element_ty))


def fused_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_len: int | torch.Tensor,
    sm_scale: float | None = None,
    block_n: int = 64,
    *,
    num_splits: int = 1,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    """Attention of one query token against ``seq_len`` cached tokens.

    ``q``: ``[H_q, D]`` fp16 (row stride arbitrary, last stride 1).
    ``k_cache`` / ``v_cache``: ``[H_kv, S_max, D]`` fp16 (last stride 1); only the
    first ``seq_len`` positions are read. Query head ``h`` attends to KV head
    ``h // (H_q / H_kv)``. Returns ``[H_q, D]`` fp16.

    ``seq_len`` is a Python int, or a one-element ``int32`` tensor on the same device
    (the CUDA-graph path; its value is not range-checked here because that would need
    a device-to-host copy, so the caller must keep ``1 <= seq_len <= S_max``).
    """
    h_q, d = q.shape
    h_kv = k_cache.shape[0]
    if h_q % h_kv:
        raise ValueError("num query heads must be a multiple of num KV heads")
    if d & (d - 1) or d < 16:
        raise ValueError("head_dim must be a power of two >= 16")
    from_ptr = isinstance(seq_len, torch.Tensor)
    if from_ptr:
        if seq_len.dtype != torch.int32 or seq_len.numel() != 1 or seq_len.device != q.device:
            raise ValueError("a tensor seq_len must be a one-element int32 tensor on the query's device")
        seqlen_ptr, seq_arg = seq_len, 0
    else:
        if not 1 <= seq_len <= k_cache.shape[1]:
            raise ValueError(f"seq_len {seq_len} outside [1, {k_cache.shape[1]}]")
        seqlen_ptr, seq_arg = q, int(seq_len)  # q is an unused placeholder pointer
    if num_splits < 1:
        raise ValueError("num_splits must be >= 1")
    if q.stride(-1) != 1 or k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
        raise ValueError("last dimension must be contiguous")
    group = h_q // h_kv
    g_pad = max(MIN_DOT_ROWS, triton.next_power_of_2(group))
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    out = torch.empty((h_q, d), dtype=q.dtype, device=q.device)
    if num_splits > 1:
        pacc = torch.empty((num_splits, h_q, d), dtype=torch.float32, device=q.device)
        pml = torch.empty((num_splits, h_q, 2), dtype=torch.float32, device=q.device)
    else:
        pacc = pml = out  # unused placeholders when NUM_SPLITS == 1
    launch = {}
    if num_warps is not None:
        launch["num_warps"] = num_warps
    if num_stages is not None:
        launch["num_stages"] = num_stages
    _decode_attn_kernel[(h_kv, num_splits)](
        q, k_cache, v_cache, out, pacc, pml, seqlen_ptr,
        seq_arg, sm_scale,
        q.stride(0), k_cache.stride(0), k_cache.stride(1),
        v_cache.stride(0), v_cache.stride(1), out.stride(0),
        h_q,
        GROUP=group, G_PAD=g_pad, HEAD_DIM=d, BLOCK_N=block_n,
        NUM_SPLITS=num_splits, SEQ_FROM_PTR=from_ptr,
        **launch,
    )
    if num_splits > 1:
        _decode_attn_merge_kernel[(h_q,)](
            pacc, pml, out, out.stride(0), h_q,
            HEAD_DIM=d, NUM_SPLITS=num_splits, NS_PAD=max(2, triton.next_power_of_2(num_splits)),
        )
    return out


def decode_attention_reference(q, k_cache, v_cache, seq_len: int, sm_scale: float | None = None):
    """fp32 PyTorch reference with the same GQA head mapping."""
    h_q, d = q.shape
    h_kv = k_cache.shape[0]
    group = h_q // h_kv
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)
    k = k_cache[:, :seq_len].float().repeat_interleave(group, dim=0)  # [H_q, S, D]
    v = v_cache[:, :seq_len].float().repeat_interleave(group, dim=0)
    scores = torch.einsum("hd,hsd->hs", q.float(), k) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hs,hsd->hd", probs, v)
