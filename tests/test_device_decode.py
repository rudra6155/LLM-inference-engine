"""Decode with the position held on the device (the CUDA-graph path) against the ordinary path.

Everything except graph capture itself runs anywhere (the Triton interpreter on CPU): the device-position
step must reproduce the Python-position step exactly, because that is the only thing a captured graph does
differently. The tests marked ``requires_cuda`` then check that capture works on a real GPU and that a
replayed graph reproduces the eager tokens.
"""

import pytest
import torch

from engine.generate import KVCache, generate
from engine.graph import DeviceDecoder, GraphCaptureError, verify_replay
from engine.model import DevicePos, EngineOptions, ModelConfig, build_synthetic_model

from .conftest import DEVICE, requires_cuda

TINY = ModelConfig(hidden_size=256, intermediate_size=512, num_layers=2, num_heads=4,
                   num_kv_heads=2, vocab_size=512, rms_norm_eps=1e-6, rope_theta=10000.0)
PROMPT = [11, 42, 7, 300, 5, 9, 80, 23]


def _model(options=None, seed=11):
    return build_synthetic_model(TINY, seed=seed, device=DEVICE, options=options, max_seq_len=64, embed_std=0.05)


def _dpos(pos: int) -> DevicePos:
    return DevicePos.from_tensor(torch.tensor([pos], dtype=torch.long, device=DEVICE))


def test_device_position_step_reproduces_the_python_position_step():
    m = _model()
    a, b = KVCache(TINY, 64, DEVICE), KVCache(TINY, 64, DEVICE)
    ids = torch.tensor(PROMPT, device=DEVICE)
    m.prefill(ids, a)
    m.prefill(ids, b)
    tok = torch.tensor(17, device=DEVICE)
    pos = len(PROMPT)
    for step in range(6):  # six consecutive positions, each fed the previous argmax
        la = m.decode_step(tok, a, pos)
        lb = m.decode_step_device(tok, b, _dpos(pos))
        assert torch.equal(la, lb), f"logits differ at step {step}: {(la.float() - lb.float()).abs().max()}"
        assert torch.equal(a.data, b.data), f"KV caches differ after step {step}"
        tok, pos = torch.argmax(la), pos + 1


def test_kv_cache_device_write_matches_int_write():
    a, b = KVCache(TINY, 16, DEVICE), KVCache(TINY, 16, DEVICE)
    k = torch.randn(1, TINY.num_kv_heads, TINY.head_dim, device=DEVICE).half()
    v = torch.randn(1, TINY.num_kv_heads, TINY.head_dim, device=DEVICE).half()
    a.write(1, 9, k, v)
    b.write(1, _dpos(9), k, v)
    assert torch.equal(a.data, b.data)
    assert not b.k(0).any() and not b.k(1)[:, :9].any() and not b.k(1)[:, 10:].any()  # only slot 9 of layer 1


def test_device_position_rejects_multi_token_input_and_reference_attention():
    m = _model()
    cache = KVCache(TINY, 64, DEVICE)
    with pytest.raises(ValueError):
        m._forward_layers(torch.zeros(2, TINY.hidden_size, device=DEVICE).half(), cache, _dpos(0))
    ref_attn = _model(EngineOptions(attention="reference"))
    with pytest.raises(ValueError, match="attention='triton'"):
        ref_attn.decode_step_device(torch.tensor(1, device=DEVICE), KVCache(TINY, 64, DEVICE), _dpos(0))


@pytest.mark.parametrize("n_new", [1, 2, 12])
def test_generation_through_the_device_decoder_matches_the_eager_loop(n_new):
    m = _model()
    cache = KVCache(TINY, 64, DEVICE)
    want = generate(m, PROMPT, n_new, ignore_eos=True).token_ids
    dec = DeviceDecoder(m, cache, use_graph=False)
    got = generate(m, PROMPT, n_new, ignore_eos=True, decoder=dec).token_ids
    assert dec.mode == "device-eager" and dec.usable
    assert got == want
    # state is reset per call: a second prompt of a different length on the same decoder / cache
    want2 = generate(m, PROMPT[:5], n_new, ignore_eos=True).token_ids
    assert generate(m, PROMPT[:5], n_new, ignore_eos=True, decoder=dec).token_ids == want2


def test_decoder_stops_at_eos_like_the_eager_loop():
    m = _model()
    first = generate(m, PROMPT, 6, ignore_eos=True).token_ids
    eos = first[2]  # stop when the third token is produced
    want = generate(m, PROMPT, 6, eos_token_ids=[eos]).token_ids
    dec = DeviceDecoder(m, KVCache(TINY, 64, DEVICE), use_graph=False)
    assert generate(m, PROMPT, 6, eos_token_ids=[eos], decoder=dec).token_ids == want


def test_failed_capture_falls_back_to_the_eager_loop_and_says_why(monkeypatch):
    m = _model()
    cache = KVCache(TINY, 64, DEVICE)
    dec = DeviceDecoder(m, cache, use_graph=True)

    def boom(self):
        raise RuntimeError("operation failed due to a previous error during capture\nsecond line")

    monkeypatch.setattr(DeviceDecoder, "_capture", boom)
    want = generate(m, PROMPT, 8, ignore_eos=True).token_ids
    got = generate(m, PROMPT, 8, ignore_eos=True, decoder=dec).token_ids
    assert dec.mode == "fallback" and not dec.usable
    assert "FALLBACK" in dec.describe() and dec.error
    assert got == want  # the eager loop ran instead


def test_graph_requested_without_cuda_is_a_recorded_fallback_not_a_crash():
    if DEVICE.type == "cuda":
        pytest.skip("this machine has CUDA")
    m = _model()
    dec = DeviceDecoder(m, KVCache(TINY, 64, DEVICE), use_graph=True)
    dec.ensure_ready()
    assert dec.mode == "fallback" and "CUDA" in dec.error


def test_decoder_is_bound_to_its_own_cache_and_validates_the_position():
    m = _model()
    dec = DeviceDecoder(m, KVCache(TINY, 64, DEVICE), use_graph=False)
    with pytest.raises(ValueError, match="own KV cache"):
        generate(m, PROMPT, 4, cache=KVCache(TINY, 64, DEVICE), decoder=dec)
    dec.ensure_ready()
    with pytest.raises(ValueError):
        dec.start(torch.tensor(1, device=DEVICE), 64)
    with pytest.raises(RuntimeError):
        DeviceDecoder(m, KVCache(TINY, 64, DEVICE)).start(torch.tensor(1, device=DEVICE), 0)  # not ready


# --------------------------------------------------------------------------- #
# The check that decides whether a captured graph is trusted                    #
# --------------------------------------------------------------------------- #
def _logits(top=20.0, runner=15.0, n=64):
    x = torch.linspace(-5.0, 5.0, n)
    x[7], x[9] = top, runner
    return x.half()


def test_verify_replay_accepts_identical_and_last_bit_different_results():
    e = _logits()
    tok = torch.tensor([7])
    verify_replay(e, tok, e.clone(), tok.clone())
    noisy = e.clone()
    noisy[7] += 0.0156  # one fp16 step at this magnitude: a different but valid cuBLAS algorithm
    noisy[3] -= 0.0156
    verify_replay(e, tok, noisy, tok.clone())


def test_verify_replay_rejects_a_frozen_position_style_error():
    e = _logits()
    wrong = e.clone()
    wrong[7], wrong[9] = e[9], e[7]  # the O(1) change a replay at the wrong position would cause
    with pytest.raises(GraphCaptureError, match="logits differ"):
        verify_replay(e, torch.tensor([7]), wrong, torch.tensor([9]))


def test_verify_replay_rejects_a_token_mismatch_with_a_clear_margin():
    e = _logits()
    with pytest.raises(GraphCaptureError, match="replay produced token 9"):
        verify_replay(e, torch.tensor([7]), e.clone(), torch.tensor([9]))  # logits agree, token does not


def test_verify_replay_tolerates_a_token_flip_inside_a_near_tie():
    e = _logits(top=20.0, runner=19.97)  # top-2 gap 0.03: rounding can flip the argmax
    verify_replay(e, torch.tensor([7]), e.clone(), torch.tensor([9]))


# --------------------------------------------------------------------------- #
# Real CUDA graphs (GPU machines only)                                         #
# --------------------------------------------------------------------------- #
@requires_cuda
@pytest.mark.parametrize("n_new", [2, 16])
def test_cuda_graph_replay_reproduces_the_eager_tokens(n_new):
    m = _model()
    cache = KVCache(TINY, 64, DEVICE)
    dec = DeviceDecoder(m, cache, use_graph=True)
    dec.ensure_ready()
    assert dec.mode == "graph", dec.describe()  # a failure here means capture does not work on this GPU
    for prompt in (PROMPT, PROMPT[:5], PROMPT + PROMPT[:4]):  # three prompt lengths on one captured graph
        want = generate(m, prompt, n_new, ignore_eos=True).token_ids
        got = generate(m, prompt, n_new, ignore_eos=True, decoder=dec).token_ids
        assert got == want, (len(prompt), got, want)


@requires_cuda
def test_cuda_graph_is_captured_once_and_replays_many_positions():
    m = _model()
    dec = DeviceDecoder(m, KVCache(TINY, 64, DEVICE), use_graph=True)
    dec.ensure_ready()
    graph = dec._graph
    generate(m, PROMPT, 10, ignore_eos=True, decoder=dec)
    generate(m, PROMPT[:3], 10, ignore_eos=True, decoder=dec)
    assert dec._graph is graph and dec.mode == "graph"
