"""Triton kernels vs PyTorch references numerical validation (atol = 1e-3).

On a machine without CUDA these run through the Triton interpreter (see
conftest.py): they prove the kernel logic and numerics, not compiled GPU code.
"""

import pytest
import torch

from engine.kernels import dequant_gemm as dg
from engine.kernels.dequant_gemm import (
    awq_gemm,
    awq_gemm_reference,
    dequantize_reference,
    pack_awq,
    unpack_awq,
)
from engine.kernels.fused_attention import decode_attention_reference, fused_decode_attention

from .conftest import DEVICE, requires_cuda
from .reference import autoawq_dequantize_gemm, random_awq_linear

ATOL = 1e-3


# --------------------------------------------------------------------------- #
# Packing / dequantization layout (checked against AutoAWQ's own code)        #
# --------------------------------------------------------------------------- #
def test_pack_matches_autoawq_packer():
    iw, iz, qw_ref, qz_ref, _ = random_awq_linear(256, 64, seed=1)
    qw, qz = pack_awq(iw, iz, 128)
    assert torch.equal(qw, qw_ref)
    assert torch.equal(qz, qz_ref)


def test_pack_unpack_roundtrip():
    iw, iz, qw, qz, _ = random_awq_linear(384, 96, seed=2)
    iw2, iz2 = unpack_awq(qw, qz, 128)
    assert torch.equal(iw2, iw)
    assert torch.equal(iz2, iz)


def test_dequantize_reference_equals_autoawq_dequantize_gemm():
    _, _, qw, qz, sc = random_awq_linear(256, 128, seed=3)
    ours = dequantize_reference(qw, qz, sc, 128)
    theirs = autoawq_dequantize_gemm(qw, qz, sc, 4, 128)
    assert ours.dtype == theirs.dtype == torch.float16
    assert torch.equal(ours, theirs)


# --------------------------------------------------------------------------- #
# Fused dequantize + GEMM                                                      #
# --------------------------------------------------------------------------- #
GEMM_SHAPES = [
    # (M, K, N): decode (M=1), tiny batch, prefill-like, N not a multiple of BLOCK_N
    # (N must be a multiple of 32: AutoAWQ's own packer requires it),
    # and the real Qwen2.5-3B down_proj reduction length K=11008 (86 groups of 128).
    (1, 256, 128),
    (1, 2048, 256),
    (1, 256, 96),
    (4, 384, 128),
    (17, 256, 128),
    (1, 11008, 128),
]


@pytest.mark.parametrize("m,k,n", GEMM_SHAPES)
@pytest.mark.parametrize("use_bias", [False, True])
def test_awq_gemm_matches_reference_fp32_out(m, k, n, use_bias):
    torch.manual_seed(10 + m + k + n)
    _, _, qw, qz, sc = random_awq_linear(k, n, seed=m + k + n)
    x = torch.randn(m, k).half()
    bias = torch.randn(n).half() if use_bias else None
    y = awq_gemm(x.to(DEVICE), qw.to(DEVICE), qz.to(DEVICE), sc.to(DEVICE),
                 None if bias is None else bias.to(DEVICE),
                 out_dtype=torch.float32)
    ref = awq_gemm_reference(x, qw, qz, sc, bias, 128)
    assert torch.allclose(y.cpu(), ref, atol=ATOL, rtol=0), (y.cpu() - ref).abs().max()


@pytest.mark.parametrize("split_k", [1, 2, 4])
def test_awq_gemm_split_k_variants_agree_with_reference(split_k):
    m, k, n = 1, 512, 128
    torch.manual_seed(5)
    _, _, qw, qz, sc = random_awq_linear(k, n, seed=5)
    x = torch.randn(m, k).half()
    cfg = {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": split_k}
    y = awq_gemm(x.to(DEVICE), qw.to(DEVICE), qz.to(DEVICE), sc.to(DEVICE),
                 out_dtype=torch.float32, config=cfg)
    ref = awq_gemm_reference(x, qw, qz, sc, None, 128)
    assert torch.allclose(y.cpu(), ref, atol=ATOL, rtol=0)


def test_awq_gemm_fp16_output():
    """fp16 outputs are rounded to 11 significant bits, so a 1-ulp flip near a
    rounding boundary is expected; tolerance is atol=1e-3 plus rtol=1e-3."""
    m, k, n = 1, 2048, 256
    torch.manual_seed(6)
    _, _, qw, qz, sc = random_awq_linear(k, n, seed=6)
    x = torch.randn(m, k).half()
    y = awq_gemm(x.to(DEVICE), qw.to(DEVICE), qz.to(DEVICE), sc.to(DEVICE))
    assert y.dtype == torch.float16
    ref = awq_gemm_reference(x, qw, qz, sc, None, 128)
    assert torch.allclose(y.cpu().float(), ref, atol=ATOL, rtol=1e-3)


def test_awq_gemm_never_calls_the_dequantize_reference(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("fused kernel must not materialize dequantized weights")

    monkeypatch.setattr(dg, "dequantize_reference", boom)
    _, _, qw, qz, sc = random_awq_linear(256, 128, seed=7)
    x = torch.randn(1, 256).half()
    awq_gemm(x.to(DEVICE), qw.to(DEVICE), qz.to(DEVICE), sc.to(DEVICE))


def test_awq_gemm_rejects_bad_shapes():
    _, _, qw, qz, sc = random_awq_linear(256, 128, seed=8)
    with pytest.raises(ValueError):
        awq_gemm(torch.randn(1, 128).half(), qw, qz, sc)  # K mismatch
    with pytest.raises(TypeError):
        awq_gemm(torch.randn(1, 256), qw, qz, sc)  # fp32 activations


@requires_cuda
def test_awq_gemm_does_not_allocate_a_dequantized_matrix_on_gpu():
    k, n = 2048, 11008
    _, _, qw, qz, sc = random_awq_linear(k, n, seed=9)
    qw, qz, sc = qw.cuda(), qz.cuda(), sc.cuda()
    x = torch.randn(1, k).half().cuda()
    awq_gemm(x, qw, qz, sc)  # warm-up / compile
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    awq_gemm(x, qw, qz, sc)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - before
    fp16_matrix_bytes = k * n * 2
    assert extra < fp16_matrix_bytes // 50, (extra, fp16_matrix_bytes)


# --------------------------------------------------------------------------- #
# Fused decode attention                                                       #
# --------------------------------------------------------------------------- #
ATTN_CASES = [
    # (H_q, H_kv, head_dim, seq_len, cache_len)
    (16, 2, 128, 1, 64),      # Qwen2.5-3B shape, single cached token
    (16, 2, 128, 17, 64),     # not a multiple of BLOCK_N
    (16, 2, 128, 64, 64),     # exactly one block
    (16, 2, 128, 200, 256),   # several blocks
    (4, 2, 64, 33, 64),       # tiny-model shape
    (8, 8, 64, 20, 64),       # MHA (group of 1)
]


@pytest.mark.parametrize("hq,hkv,d,s,cache_len", ATTN_CASES)
def test_decode_attention_matches_reference(hq, hkv, d, s, cache_len):
    torch.manual_seed(100 + s)
    q = (torch.randn(hq, d) * 2).half()
    kc = torch.randn(hkv, cache_len, d).half()
    vc = torch.randn(hkv, cache_len, d).half()
    out = fused_decode_attention(q.to(DEVICE), kc.to(DEVICE), vc.to(DEVICE), s)
    ref = decode_attention_reference(q, kc, vc, s)
    assert out.dtype == torch.float16
    assert torch.allclose(out.cpu().float(), ref, atol=ATOL, rtol=0), (out.cpu().float() - ref).abs().max()


def test_decode_attention_reads_only_the_first_seq_len_positions():
    torch.manual_seed(1)
    q = torch.randn(16, 128).half()
    kc = torch.randn(2, 64, 128).half()
    vc = torch.randn(2, 64, 128).half()
    a = fused_decode_attention(q.to(DEVICE), kc.to(DEVICE), vc.to(DEVICE), 20)
    kc2, vc2 = kc.clone(), vc.clone()
    kc2[:, 20:] = 1e4  # garbage beyond seq_len must not matter
    vc2[:, 20:] = -1e4
    b = fused_decode_attention(q.to(DEVICE), kc2.to(DEVICE), vc2.to(DEVICE), 20)
    assert torch.equal(a.cpu(), b.cpu())


def test_decode_attention_accepts_strided_query():
    torch.manual_seed(2)
    big = torch.randn(3, 16, 128).half()
    q = big[1]  # a view with a non-trivial storage offset
    kc = torch.randn(2, 32, 128).half()
    vc = torch.randn(2, 32, 128).half()
    out = fused_decode_attention(q.to(DEVICE), kc.to(DEVICE), vc.to(DEVICE), 32)
    ref = decode_attention_reference(q, kc, vc, 32)
    assert torch.allclose(out.cpu().float(), ref, atol=ATOL, rtol=0)


def test_decode_attention_rejects_bad_arguments():
    q = torch.randn(16, 128).half()
    kc = torch.randn(2, 32, 128).half()
    with pytest.raises(ValueError):
        fused_decode_attention(q, kc, kc, 0)
    with pytest.raises(ValueError):
        fused_decode_attention(q, kc, kc, 33)
    with pytest.raises(ValueError):
        fused_decode_attention(torch.randn(15, 128).half(), kc, kc, 8)
