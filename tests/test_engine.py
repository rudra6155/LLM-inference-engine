"""Unit and integration tests: KV-cache memory model and end-to-end
generation against references.

Tiny-model tests use random weights in the real AWQ layout and run anywhere. The
real-checkpoint test runs on a CUDA machine by default, and on CPU only with
RUN_SLOW=1 (the Triton interpreter makes it take a long time).
"""

import json
import os
from pathlib import Path

import pytest
import torch

import engine.model as model_mod
from engine.generate import KVCache, generate
from engine.model import (
    QWEN2_5_3B,
    EngineOptions,
    ModelConfig,
    build_synthetic_model,
    dtype_bytes,
    kv_cache_bytes_per_token,
)

from .conftest import DEVICE, HAS_CUDA, requires_slow
from .reference import build_hf_from_engine_model, build_hf_reference

TINY = ModelConfig(hidden_size=256, intermediate_size=512, num_layers=2, num_heads=4,
                   num_kv_heads=2, vocab_size=512, rms_norm_eps=1e-6, rope_theta=10000.0)
REFERENCE_OPTIONS = EngineOptions(gemm="reference", attention="reference", rmsnorm="reference")
PROMPT = [11, 42, 7, 300, 5]


# --------------------------------------------------------------------------- #
# KV-cache memory model                                                        #
# --------------------------------------------------------------------------- #
def test_qwen2_5_3b_kv_bytes_per_token_is_36864():
    # 2 (K and V) * 36 layers * 2 kv heads * 128 head_dim * 2 bytes (fp16)
    assert QWEN2_5_3B.head_dim == 128
    assert kv_cache_bytes_per_token(QWEN2_5_3B, torch.float16) == 2 * 36 * 2 * 128 * 2 == 36864


@pytest.mark.parametrize("cfg", [QWEN2_5_3B, TINY])
@pytest.mark.parametrize("max_seq", [1, 64, 300])
def test_allocated_kv_cache_matches_analytic_formula(cfg, max_seq):
    cache = KVCache(cfg, max_seq, "cpu")
    analytic = 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * dtype_bytes(torch.float16) * max_seq
    assert cache.nbytes == analytic
    assert cache.data.untyped_storage().nbytes() == analytic  # bytes actually allocated
    assert cache.bytes_per_token * max_seq == analytic


def test_kv_cache_bytes_scale_with_dtype():
    assert kv_cache_bytes_per_token(QWEN2_5_3B, torch.float32) == 2 * kv_cache_bytes_per_token(QWEN2_5_3B, torch.float16)


def test_kv_cache_write_and_read_back():
    cache = KVCache(TINY, 16, "cpu")
    k = torch.randn(3, TINY.num_kv_heads, TINY.head_dim).half()
    v = torch.randn(3, TINY.num_kv_heads, TINY.head_dim).half()
    cache.write(1, 4, k, v)
    assert torch.equal(cache.k(1)[:, 4:7], k.transpose(0, 1))
    assert torch.equal(cache.v(1)[:, 4:7], v.transpose(0, 1))
    assert not cache.k(0).any()  # other layers untouched
    with pytest.raises(ValueError):
        cache.write(0, 15, k, v)  # overflow


def test_config_from_json_reads_real_checkpoint_values(tmp_path):
    cfg_json = {
        "hidden_size": 2048, "intermediate_size": 11008, "num_hidden_layers": 36,
        "num_attention_heads": 16, "num_key_value_heads": 2, "vocab_size": 151936,
        "rms_norm_eps": 1e-06, "rope_theta": 1000000.0, "tie_word_embeddings": True,
        "quantization_config": {"bits": 4, "group_size": 128, "quant_method": "awq",
                                "version": "gemm", "zero_point": True},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg_json))
    assert ModelConfig.from_json(path) == QWEN2_5_3B
    cfg_json["quantization_config"]["version"] = "gemv"
    path.write_text(json.dumps(cfg_json))
    with pytest.raises(ValueError):
        ModelConfig.from_json(path)


# --------------------------------------------------------------------------- #
# Tiny-model end-to-end                                                        #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tiny_models():
    # seed 11 was chosen (scan over seeds 0-13) for the largest worst-case top-2 logit margin
    fast = build_synthetic_model(TINY, seed=11, device=DEVICE, max_seq_len=64)  # triton + C++
    ref = fast.with_options(REFERENCE_OPTIONS)  # same weights, pure PyTorch ops
    return fast, ref


def _first_token(model, ids):
    cache = KVCache(model.cfg, model.max_seq_len, model.device)
    return generate(model, ids, 1, cache=cache).token_ids[0]


def _min_top2_gap(model, prompt, n_tokens):
    """Smallest top-1 minus top-2 logit gap along the model's own greedy path.

    An exact-match test is only meaningful when the winner beats the runner-up by
    more than fp16 noise (observed fast-vs-reference logit error is ~2e-3), so the
    tests assert this margin on their data instead of silently risking a near-tie.
    """
    cache = KVCache(model.cfg, model.max_seq_len, model.device)
    logits = model.prefill(torch.tensor(prompt, device=model.device), cache)
    gaps, pos = [], len(prompt)
    for _ in range(n_tokens):
        top = logits.float().topk(2)
        gaps.append((top.values[0] - top.values[1]).item())
        logits = model.decode_step(top.indices[0], cache, pos)
        pos += 1
    return min(gaps)


MATCH_PROMPTS = [PROMPT, [3], [100, 200, 300, 400, 500, 1, 2, 3, 4]]
MIN_MARGIN = 0.1


@pytest.mark.parametrize("prompt", MATCH_PROMPTS)
def test_first_token_matches_reference_exactly(tiny_models, prompt):
    fast, ref = tiny_models
    assert _min_top2_gap(ref, prompt, 1) > MIN_MARGIN  # test data is not a near-tie
    assert _first_token(fast, prompt) == _first_token(ref, prompt)


def test_generated_sequence_matches_reference(tiny_models):
    fast, ref = tiny_models
    assert _min_top2_gap(ref, PROMPT, 8) > MIN_MARGIN
    a = generate(fast, PROMPT, 8, ignore_eos=True).token_ids
    b = generate(ref, PROMPT, 8, ignore_eos=True).token_ids
    assert a == b and len(a) == 8


@pytest.fixture(scope="module")
def tiny_hf(tiny_models):
    """HuggingFace Qwen2 built from the same dequantized weights: an implementation of
    the architecture (RoPE, GQA, RMSNorm, biases, SwiGLU) that shares no code with ours."""
    pytest.importorskip("transformers")
    return build_hf_from_engine_model(tiny_models[0])


def _hf_logits(hf, prompt):
    with torch.no_grad():
        return hf(torch.tensor([prompt])).logits[0, -1].float()


@pytest.mark.parametrize("prompt", MATCH_PROMPTS)
def test_architecture_matches_huggingface_qwen2(tiny_models, tiny_hf, prompt):
    fast, ref = tiny_models
    hf = _hf_logits(tiny_hf, prompt)
    ours_ref = ref.prefill(torch.tensor(prompt, device=ref.device),
                           KVCache(ref.cfg, ref.max_seq_len, ref.device)).float().cpu()
    ours_fast = fast.prefill(torch.tensor(prompt, device=fast.device),
                             KVCache(fast.cfg, fast.max_seq_len, fast.device)).float().cpu()
    top = hf.topk(2)
    assert (top.values[0] - top.values[1]) > MIN_MARGIN
    assert int(ours_ref.argmax()) == int(ours_fast.argmax()) == int(hf.argmax())
    # Measured max |difference| to HF is <= 2e-3 (fp16 rounding); a wrong RoPE theta gives
    # >= 2e-2, so 1e-2 separates a correct implementation from plausible bugs.
    assert torch.allclose(ours_ref, hf, atol=1e-2, rtol=0), (ours_ref - hf).abs().max()
    assert torch.allclose(ours_fast, hf, atol=1e-2, rtol=0), (ours_fast - hf).abs().max()


def test_long_context_logits_match_huggingface_qwen2(tiny_models, tiny_hf):
    """40 tokens amplify position-dependent mistakes (RoPE frequencies, cache indexing)."""
    fast, ref = tiny_models
    prompt = list(range(5, 45))
    hf = _hf_logits(tiny_hf, prompt)
    for m in (ref, fast):
        ours = m.prefill(torch.tensor(prompt, device=m.device), KVCache(m.cfg, m.max_seq_len, m.device))
        assert torch.allclose(ours.float().cpu(), hf, atol=1e-2, rtol=0), (ours.float().cpu() - hf).abs().max()


def test_greedy_sequence_matches_huggingface_qwen2(tiny_models, tiny_hf):
    """Eight greedy steps, re-running the full HF forward each time (not generate())."""
    fast, _ = tiny_models
    ids = list(PROMPT)
    expected, margins = [], []
    for _ in range(8):
        logits = _hf_logits(tiny_hf, ids)
        top = logits.topk(2)
        margins.append((top.values[0] - top.values[1]).item())
        ids.append(int(top.indices[0]))
        expected.append(ids[-1])
    assert min(margins) > MIN_MARGIN
    assert generate(fast, PROMPT, 8, ignore_eos=True).token_ids == expected


def test_output_depends_on_earlier_tokens_through_attention(tiny_models):
    """Wiring check: two prompts that differ only in their first token must give
    different last-position logits, otherwise attention / the KV cache is not
    influencing the output and the exact-match tests above would prove little."""
    _, ref = tiny_models
    a = ref.prefill(torch.tensor([11, 42, 7, 300, 5], device=ref.device),
                    KVCache(ref.cfg, ref.max_seq_len, ref.device)).float()
    b = ref.prefill(torch.tensor([12, 42, 7, 300, 5], device=ref.device),
                    KVCache(ref.cfg, ref.max_seq_len, ref.device)).float()
    assert (a - b).abs().max() > 0.05


def test_decode_with_kv_cache_equals_full_prefill(tiny_models):
    """Logits after prefill(n tokens) + decode(token n) must equal prefill(n+1 tokens):
    this exercises RoPE positions and cache reads/writes."""
    _, ref = tiny_models
    ids = torch.tensor(PROMPT + [77], device=ref.device)
    full_cache = KVCache(ref.cfg, ref.max_seq_len, ref.device)
    full = ref.prefill(ids, full_cache)

    inc_cache = KVCache(ref.cfg, ref.max_seq_len, ref.device)
    ref.prefill(ids[:-1], inc_cache)
    inc = ref.decode_step(ids[-1], inc_cache, len(PROMPT))
    assert torch.allclose(full.float(), inc.float(), atol=2e-2, rtol=0), (full.float() - inc.float()).abs().max()


def test_fast_path_logits_close_to_reference_logits(tiny_models):
    fast, ref = tiny_models
    ids = torch.tensor(PROMPT, device=fast.device)
    a = fast.prefill(ids, KVCache(fast.cfg, fast.max_seq_len, fast.device)).float()
    b = ref.prefill(ids, KVCache(ref.cfg, ref.max_seq_len, ref.device)).float()
    assert torch.allclose(a, b, atol=5e-2, rtol=0), (a - b).abs().max()


def test_default_execution_path_never_materializes_weights(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("default path must not dequantize weights")

    monkeypatch.setattr(model_mod, "dequantize_reference", boom)
    model = build_synthetic_model(TINY, seed=1, device=DEVICE, max_seq_len=32)
    generate(model, PROMPT, 3, ignore_eos=True)


def test_generation_stops_at_eos_unless_ignored(tiny_models):
    fast, _ = tiny_models
    first = generate(fast, PROMPT, 1).token_ids[0]
    stopped = generate(fast, PROMPT, 6, eos_token_ids=[first])
    assert stopped.token_ids == [first]
    forced = generate(fast, PROMPT, 6, eos_token_ids=[first], ignore_eos=True)
    assert len(forced.token_ids) == 6


def test_generation_reports_timings(tiny_models):
    fast, _ = tiny_models
    r = generate(fast, PROMPT, 5, ignore_eos=True)
    assert r.prompt_tokens == len(PROMPT)
    assert r.ttft_s > 0 and len(r.itl_s) == 4 and all(t > 0 for t in r.itl_s)


def test_generation_rejects_overlong_requests(tiny_models):
    fast, _ = tiny_models
    with pytest.raises(ValueError):
        generate(fast, PROMPT, fast.max_seq_len)  # prompt + new tokens > max_seq_len
    with pytest.raises(ValueError):
        generate(fast, [], 4)


def test_primary_path_does_not_use_huggingface_generate():
    src = Path(model_mod.__file__).parent
    for name in ("generate.py", "model.py"):
        text = (src / name).read_text()
        assert "from transformers" not in text and "import transformers" not in text
        assert ".generate(" not in text.replace("def generate(", "")


# --------------------------------------------------------------------------- #
# safetensors loader (the explicit, non-autoawq path)                          #
# --------------------------------------------------------------------------- #
def _write_tiny_checkpoint(model, directory, lm_head_differs=False):
    from safetensors.torch import save_file

    cfg = model.cfg
    tensors = {"model.embed_tokens.weight": model.embed.cpu(), "model.norm.weight": model.final_norm.cpu()}
    lm = model.embed.cpu().clone()
    if lm_head_differs:
        lm[0, 0] += 1
    tensors["lm_head.weight"] = lm
    for i, lw in enumerate(model.layers):
        pre = f"model.layers.{i}."
        for short, attr in model_mod._PROJ.items():
            lin = getattr(lw, short)
            tensors[pre + attr + ".qweight"] = lin.qweight.cpu().contiguous()
            tensors[pre + attr + ".qzeros"] = lin.qzeros.cpu().contiguous()
            tensors[pre + attr + ".scales"] = lin.scales.cpu().contiguous()
            if lin.bias is not None:
                tensors[pre + attr + ".bias"] = lin.bias.cpu().contiguous()
        tensors[pre + "input_layernorm.weight"] = lw.input_ln.cpu()
        tensors[pre + "post_attention_layernorm.weight"] = lw.post_ln.cpu()
    save_file(tensors, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps({
        "hidden_size": cfg.hidden_size, "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.num_layers, "num_attention_heads": cfg.num_heads,
        "num_key_value_heads": cfg.num_kv_heads, "vocab_size": cfg.vocab_size,
        "rms_norm_eps": cfg.rms_norm_eps, "rope_theta": cfg.rope_theta, "tie_word_embeddings": True,
        "quantization_config": {"bits": 4, "group_size": cfg.group_size, "quant_method": "awq",
                                "version": "gemm", "zero_point": True},
    }))


def test_safetensors_loader_round_trips_a_checkpoint(tiny_models, tmp_path):
    pytest.importorskip("safetensors")
    fast, _ = tiny_models
    _write_tiny_checkpoint(fast, tmp_path)
    loaded = model_mod.load_awq_model(str(tmp_path), loader="safetensors", device=DEVICE, max_seq_len=64)
    assert loaded.cfg == fast.cfg
    assert torch.equal(loaded.embed.cpu(), fast.embed.cpu())
    for a, b in zip(loaded.layers, fast.layers):
        for short in model_mod._PROJ:
            la, lb = getattr(a, short), getattr(b, short)
            assert torch.equal(la.qweight.cpu(), lb.qweight.cpu())
            assert torch.equal(la.qzeros.cpu(), lb.qzeros.cpu())
            assert torch.equal(la.scales.cpu(), lb.scales.cpu())
            assert (la.bias is None) == (lb.bias is None)
    assert loaded.embed.device.type == DEVICE.type  # tensors ended up on the requested device
    ids = torch.tensor(PROMPT, device=DEVICE)
    got = loaded.prefill(ids, KVCache(loaded.cfg, 64, loaded.device))
    want = fast.prefill(ids, KVCache(fast.cfg, 64, fast.device))
    assert torch.equal(got, want)


def test_safetensors_loader_refuses_untied_embeddings_when_config_says_tied(tiny_models, tmp_path):
    pytest.importorskip("safetensors")
    _write_tiny_checkpoint(tiny_models[0], tmp_path, lm_head_differs=True)
    with pytest.raises(ValueError, match="lm_head"):
        model_mod.load_awq_model(str(tmp_path), loader="safetensors", device=DEVICE, max_seq_len=64)


def test_autoawq_loader_is_never_selected_implicitly(tmp_path):
    with pytest.raises(ValueError, match="unknown loader"):
        model_mod.load_awq_model(str(tmp_path), loader="auto")


# --------------------------------------------------------------------------- #
# Real checkpoint                                                              #
# --------------------------------------------------------------------------- #
@requires_slow
def test_real_checkpoint_first_token_matches_reference():
    """Verify the first generated token equals the HuggingFace reference implementation.

    Reference: HuggingFace ``Qwen2ForCausalLM`` (its own attention / RoPE / norm code)
    loaded with the AutoAWQ-dequantized checkpoint weights, used for a single
    forward pass (not ``generate()``). The reference holds the full fp16 weights in host
    RAM - acceptable for a test, never done by the engine. Measured peak host RSS of building
    and running the reference: 7.5 GiB. The first run downloads ~2 GB.
    """
    from transformers import AutoTokenizer

    name = os.environ.get("ENGINE_MODEL", "Qwen/Qwen2.5-3B-Instruct-AWQ")
    model_dir = model_mod.resolve_model_dir(name)
    hf_cfg = json.loads((model_dir / "config.json").read_text())
    assert ModelConfig.from_json(model_dir / "config.json") == QWEN2_5_3B

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    prompt_ids = tok("The capital of France is", return_tensors="pt").input_ids[0].tolist()

    # ENGINE_LOADER=safetensors is the fallback if autoawq is unavailable;
    # the loader used is printed for traceability.
    loader = os.environ.get("ENGINE_LOADER") or ("autoawq" if HAS_CUDA else "safetensors")
    engine = model_mod.load_awq_model(str(model_dir), loader=loader, device=DEVICE, max_seq_len=64)
    got = generate(engine, prompt_ids, 1).token_ids[0]

    ref = build_hf_reference(model_dir, {k: v for k, v in hf_cfg.items() if k != "quantization_config"})
    with torch.no_grad():
        logits = ref(torch.tensor([prompt_ids])).logits[0, -1]
    expected = int(torch.argmax(logits))
    print(f"engine loader={loader} first token {got!r}; reference {expected!r}")
    assert got == expected
