"""Tuned-config file handling and the GEMM launch options it feeds.

The tuner itself needs a GPU to produce meaningful timings; here we check that what it writes is validated on
load, that a tuned config is applied only where it should be (decode, M == 1), and that tuned launch options
do not change results.
"""

import json

import pytest
import torch

from engine.generate import KVCache
from engine.kernels.dequant_gemm import awq_gemm, awq_gemm_reference, default_config, validate_config
from engine.kernels.tuning import SCHEMA, TunedConfigs, gemm_key, validate_attention
from engine.model import EngineOptions, ModelConfig, build_synthetic_model

from .conftest import DEVICE
from .reference import random_awq_linear

ATOL = 1e-3
TINY = ModelConfig(hidden_size=256, intermediate_size=512, num_layers=2, num_heads=4,
                   num_kv_heads=2, vocab_size=512, rms_norm_eps=1e-6, rope_theta=10000.0)
GOOD_GEMM = {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 2, "num_warps": 4, "num_stages": 2}


def _doc(**over):
    d = {"schema": SCHEMA, "meta": {"device": "test"}, "awq_gemm": {"256x256": GOOD_GEMM},
         "attention": {"block_n": 32, "num_warps": 2, "num_stages": 2, "num_splits": 2}}
    d.update(over)
    return d


# ---- GEMM launch options -------------------------------------------------- #
@pytest.mark.parametrize("cfg", [
    {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 2, "num_warps": 8, "num_stages": 2},
    {"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 4},
    {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 1, "num_warps": 2},
])
def test_awq_gemm_with_tuned_launch_options_matches_the_reference(cfg):
    k, n = 512, 256
    _, _, qw, qz, sc = random_awq_linear(k, n, seed=3)
    x = torch.randn(1, k).half()
    y = awq_gemm(x.to(DEVICE), qw.to(DEVICE), qz.to(DEVICE), sc.to(DEVICE), out_dtype=torch.float32, config=cfg)
    assert torch.allclose(y.cpu(), awq_gemm_reference(x, qw, qz, sc, None, 128), atol=ATOL, rtol=0)


@pytest.mark.parametrize("bad", [
    {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128},                                   # missing SPLIT_K
    {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 1, "BLOCK_Q": 3},       # unknown key
    {"BLOCK_M": 16, "BLOCK_N": 60, "BLOCK_K": 128, "SPLIT_K": 1},                     # BLOCK_N not a multiple of 8
    {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 96, "SPLIT_K": 1},                      # BLOCK_K does not divide the group
    {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 8},                     # more splits than K-blocks (K=512 -> 4)
    {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 0},
    {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 1, "num_warps": True},  # a bool is not a count
])
def test_validate_config_rejects_bad_configs(bad):
    with pytest.raises(ValueError):
        validate_config(bad, 512, 128)


def test_awq_gemm_refuses_an_invalid_config_instead_of_launching_it():
    _, _, qw, qz, sc = random_awq_linear(512, 256, seed=3)
    x = torch.randn(1, 512).half().to(DEVICE)
    with pytest.raises(ValueError):
        awq_gemm(x, qw.to(DEVICE), qz.to(DEVICE), sc.to(DEVICE), config={"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 99})


def test_default_config_is_unchanged_by_the_new_launch_options():
    cfg = default_config(1, 2048, 2048, 128, torch.device("cpu"))
    assert set(cfg) == {"BLOCK_M", "BLOCK_N", "BLOCK_K", "SPLIT_K"}  # no num_warps / num_stages: Triton defaults


# ---- the tuned-config file ------------------------------------------------ #
def test_a_valid_file_round_trips(tmp_path):
    t = TunedConfigs.from_dict(_doc(), group_size=128)
    path = tmp_path / "best.json"
    t.save(path)
    again = TunedConfigs.load(path)
    assert again.gemm == t.gemm == {(256, 256): GOOD_GEMM}
    assert again.attention == {"block_n": 32, "num_warps": 2, "num_stages": 2, "num_splits": 2}
    assert json.loads(path.read_text())["awq_gemm"] == {gemm_key(256, 256): GOOD_GEMM}


def test_tuned_gemm_config_applies_to_decode_shapes_only():
    t = TunedConfigs.from_dict(_doc())
    assert t.gemm_config(1, 256, 256) == GOOD_GEMM
    assert t.gemm_config(2, 256, 256) is None      # prefill-shaped (M > 1): the default heuristic
    assert t.gemm_config(1, 256, 999) is None      # a shape that was not tuned


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.update(schema=2), "schema"),
    (lambda d: d["awq_gemm"].update({"256-256": GOOD_GEMM}), "KxN"),
    (lambda d: d["awq_gemm"].update({"256x256": {**GOOD_GEMM, "BLOCK_K": 96}}), "BLOCK_K"),
    (lambda d: d["awq_gemm"].update({"256x256": {**GOOD_GEMM, "SPLIT_K": 8}}), "SPLIT_K"),
    (lambda d: d["attention"].update(block_n=48), "power of two"),
    (lambda d: d["attention"].update(num_gpus=1), "unknown keys"),
    (lambda d: d["attention"].update(num_splits=0), "positive"),
])
def test_invalid_files_fail_on_load(mutate, match):
    d = _doc()
    mutate(d)
    with pytest.raises(ValueError, match=match):
        TunedConfigs.from_dict(d)


def test_missing_file_names_the_tuner(tmp_path):
    with pytest.raises(FileNotFoundError, match="tune_kernels"):
        TunedConfigs.load(tmp_path / "nope.json")


def test_attention_validation_accepts_the_defaults_and_an_empty_config():
    assert validate_attention({}) == {}
    assert validate_attention({"block_n": 64, "num_warps": 4, "num_stages": 3, "num_splits": 1})


# ---- end to end: a tuned model ------------------------------------------- #
def test_a_model_with_a_tuned_config_computes_the_same_logits_up_to_accumulation_order(tmp_path):
    shapes = {(256, 256): GOOD_GEMM,                                                   # q, o
              (256, 128): {**GOOD_GEMM, "BLOCK_N": 64, "SPLIT_K": 4, "BLOCK_K": 64},  # k, v (2 kv heads x 64)
              (256, 512): {**GOOD_GEMM, "BLOCK_N": 128, "SPLIT_K": 1, "BLOCK_K": 64},    # gate, up
              (512, 256): {**GOOD_GEMM, "BLOCK_N": 64, "SPLIT_K": 2, "BLOCK_K": 128}}   # down
    doc = _doc(awq_gemm={gemm_key(k, n): c for (k, n), c in shapes.items()})
    path = tmp_path / "tuned.json"
    path.write_text(json.dumps(doc))

    plain = build_synthetic_model(TINY, seed=11, device=DEVICE, max_seq_len=64, embed_std=0.05)
    tuned = build_synthetic_model(TINY, seed=11, device=DEVICE, max_seq_len=64, embed_std=0.05,
                                  options=EngineOptions(tuned_config=str(path)))
    assert tuned.tuned is not None and tuned.tuned.gemm.keys() == shapes.keys()
    ids = torch.tensor([11, 42, 7, 300, 5, 9], device=DEVICE)
    ca, cb = KVCache(TINY, 64, DEVICE), KVCache(TINY, 64, DEVICE)
    la, lb = plain.prefill(ids, ca), tuned.prefill(ids, cb)      # prefill (M > 1): untuned path, attention unaffected
    assert torch.allclose(la.float(), lb.float(), atol=2e-2)
    tok = torch.argmax(la)
    da, db = plain.decode_step(tok, ca, 6), tuned.decode_step(tok, cb, 6)  # decode (M == 1): tuned GEMMs + split-KV attention
    assert torch.allclose(da.float(), db.float(), atol=2e-2), (da.float() - db.float()).abs().max()
