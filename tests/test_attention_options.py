"""Decode-attention options added for CUDA graphs and for using more of the GPU:
``seq_len`` as a device tensor, split-KV (``num_splits``), and Triton launch options.

Without CUDA these run through the Triton interpreter (kernel logic and numerics only).
"""

import pytest
import torch

from engine.kernels.fused_attention import decode_attention_reference, fused_decode_attention

from .conftest import DEVICE

ATOL = 1e-3


def _case(seed, hq=16, hkv=2, d=128, cache_len=256):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(hq, d, generator=g) * 2).half()
    kc = torch.randn(hkv, cache_len, d, generator=g).half()
    vc = torch.randn(hkv, cache_len, d, generator=g).half()
    return q, kc, vc


def _run(q, kc, vc, s, **kw):
    return fused_decode_attention(q.to(DEVICE), kc.to(DEVICE), vc.to(DEVICE), s, **kw).cpu().float()


@pytest.mark.parametrize("s", [1, 17, 64, 200])
def test_tensor_seq_len_equals_python_seq_len(s):
    q, kc, vc = _case(10 + s)
    t = torch.tensor([s], dtype=torch.int32, device=DEVICE)
    a, b = _run(q, kc, vc, s), _run(q, kc, vc, t)
    assert torch.equal(a, b)  # same arithmetic, only the source of the length differs
    assert torch.allclose(b, decode_attention_reference(q, kc, vc, s), atol=ATOL, rtol=0)


def test_tensor_seq_len_is_read_at_run_time_not_baked_in():
    """What a CUDA graph needs: the same launch, a different length in memory."""
    q, kc, vc = _case(3)
    t = torch.tensor([5], dtype=torch.int32, device=DEVICE)
    for s in (5, 40, 130, 1):
        t.fill_(s)  # in-place update, as a replayed graph sees it
        out = _run(q, kc, vc, t)
        assert torch.allclose(out, decode_attention_reference(q, kc, vc, s), atol=ATOL, rtol=0), s


@pytest.mark.parametrize("num_splits", [2, 3, 4, 8])
@pytest.mark.parametrize("s", [1, 17, 64, 200])
def test_split_kv_matches_reference(num_splits, s):
    # s=1 with 8 splits leaves most slices empty; s=17 leaves some slices shorter than BLOCK_N.
    q, kc, vc = _case(50 + s)
    out = _run(q, kc, vc, s, num_splits=num_splits)
    ref = decode_attention_reference(q, kc, vc, s)
    assert torch.allclose(out, ref, atol=ATOL, rtol=0), (out - ref).abs().max()


def test_split_kv_with_tensor_seq_len_and_other_shapes():
    for hq, hkv, d in ((4, 2, 64), (8, 8, 64)):  # tiny-model GQA and MHA (group of 1)
        q, kc, vc = _case(7, hq=hq, hkv=hkv, d=d, cache_len=128)
        t = torch.tensor([77], dtype=torch.int32, device=DEVICE)
        out = _run(q, kc, vc, t, num_splits=4)
        assert torch.allclose(out, decode_attention_reference(q, kc, vc, 77), atol=ATOL, rtol=0)


def test_split_kv_ignores_positions_beyond_seq_len():
    q, kc, vc = _case(9)
    a = _run(q, kc, vc, 90, num_splits=4)
    kc2, vc2 = kc.clone(), vc.clone()
    kc2[:, 90:] = 1e4
    vc2[:, 90:] = -1e4
    assert torch.equal(a, _run(q, kc2, vc2, 90, num_splits=4))


# block_n = 128 can need more shared memory than a block on an RTX 4060 is allowed (by my estimate from the tile
# sizes; the interpreter has no such limit), so it is not a unit-test case; the tuner records a candidate that
# cannot launch as invalid instead.
@pytest.mark.parametrize("block_n", [32, 64])
def test_block_n_and_launch_options_do_not_change_the_result(block_n):
    q, kc, vc = _case(21)
    ref = decode_attention_reference(q, kc, vc, 150)
    out = _run(q, kc, vc, 150, block_n=block_n, num_warps=2, num_stages=2, num_splits=2)
    assert torch.allclose(out, ref, atol=ATOL, rtol=0)


def test_bad_tensor_seq_len_and_num_splits_are_rejected():
    q, kc, vc = _case(1, cache_len=64)
    for bad in (torch.tensor([5], dtype=torch.int64), torch.tensor([5, 6], dtype=torch.int32)):
        with pytest.raises(ValueError):
            fused_decode_attention(q.to(DEVICE), kc.to(DEVICE), vc.to(DEVICE), bad.to(DEVICE))
    with pytest.raises(ValueError):
        fused_decode_attention(q.to(DEVICE), kc.to(DEVICE), vc.to(DEVICE), 8, num_splits=0)
