"""The roofline's byte model must equal the bytes of the tensors it describes."""

import importlib.util
from pathlib import Path

from engine.model import QWEN2_5_3B, ModelConfig, build_synthetic_model

from .conftest import DEVICE

_spec = importlib.util.spec_from_file_location("roofline", Path(__file__).resolve().parents[1] / "scripts" / "roofline.py")
roofline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(roofline)

TINY = ModelConfig(256, 512, 2, 4, 2, 512, 1e-6, 10000.0)


def _layer_weight_bytes(cfg):
    h, i, g, kv = cfg.hidden_size, cfg.intermediate_size, cfg.group_size, cfg.num_kv_heads * cfg.head_dim
    lw = roofline.linear_weight_bytes
    return (lw(h, h, g, True) + lw(h, kv, g, True) + lw(h, kv, g, True) + lw(h, h, g, False)
            + 2 * lw(h, i, g, False) + lw(i, h, g, False))


def test_linear_weight_bytes_equal_synthetic_model_tensor_bytes():
    model = build_synthetic_model(TINY, seed=0, device=DEVICE, max_seq_len=16)
    assert TINY.num_layers * _layer_weight_bytes(TINY) == model.weight_bytes()["quantized_linears"]


def test_linear_weight_bytes_equal_real_checkpoint_tensor_bytes():
    # Sum of every qweight/qzeros/scales/bias tensor in Qwen2.5-3B-Instruct-AWQ's safetensors
    # header (read 2026-10-29): 1,441,640,448 bytes.
    assert QWEN2_5_3B.num_layers * _layer_weight_bytes(QWEN2_5_3B) == 1_441_640_448


def test_decode_is_memory_bound_by_a_wide_margin():
    m = roofline.analytic_model(QWEN2_5_3B, 512)
    lo_peak_gflops = m["ceilings"]["fp32_nontensor_peak_GFLOPs_range"][0]
    ridge_low = lo_peak_gflops / roofline.DATASHEET_BW_GBS  # FLOP/B at the lowest compute ceiling
    assert all(op["ai"] < ridge_low / 4 for op in m["ops"] if op["category"] != "decode attention")
    assert m["per_token"]["ai"] < ridge_low / 10


def test_lm_head_is_the_largest_single_fp16_read():
    m = roofline.analytic_model(QWEN2_5_3B, 512)
    lm = next(o for o in m["ops"] if o["name"] == "lm_head")
    assert lm["bytes"] > 622_000_000 and abs(lm["ai"] - 1.0) < 1e-2
