"""Independent references for the tests.

``autoawq_*`` functions are ported from AutoAWQ 0.2.9 (MIT licence):
``WQLinear_GEMM.from_linear`` (packing loop) and ``awq/utils/packing_utils.py``
(``unpack_awq`` / ``reverse_awq_order`` / ``dequantize_gemm``). They are kept
verbatim in logic so the engine's own packing / dequantization code is checked
against the library that produced the checkpoint, not against itself.
"""

from __future__ import annotations

import torch

AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def autoawq_pack(intweight: torch.Tensor, w_bit: int = 4) -> torch.Tensor:
    """``intweight``: int32 ``[rows, N]`` -> packed int32 ``[rows, N // 8]``."""
    pack_num = 32 // w_bit
    packed = torch.zeros((intweight.shape[0], intweight.shape[1] // 32 * w_bit), dtype=torch.int32)
    for col in range(intweight.shape[1] // pack_num):
        order_map = [0, 2, 4, 6, 1, 3, 5, 7]
        for i in range(pack_num):
            packed_col = intweight[:, col * pack_num + order_map[i]]
            packed[:, col] |= packed_col << (i * w_bit)
    return packed


def _unpack_awq(qweight, qzeros, bits):
    shifts = torch.arange(0, 32, bits, device=qzeros.device)
    iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(torch.int8)
    iweights = iweights.view(iweights.shape[0], -1)
    izeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(torch.int8)
    izeros = izeros.view(izeros.shape[0], -1)
    return iweights, izeros


def _reverse_awq_order(iweights, izeros, bits):
    reverse_order_tensor = torch.arange(iweights.shape[-1], dtype=torch.int32, device=izeros.device)
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)
    return iweights[:, reverse_order_tensor], izeros[:, reverse_order_tensor]


def autoawq_dequantize_gemm(qweight, qzeros, scales, bits, group_size):
    iweight, izeros = _unpack_awq(qweight, qzeros, bits)
    iweight, izeros = _reverse_awq_order(iweight, izeros, bits)
    iweight = torch.bitwise_and(iweight, (2**bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)
    scales = scales.repeat_interleave(group_size, dim=0)
    izeros = izeros.repeat_interleave(group_size, dim=0)
    return (iweight - izeros) * scales


def random_awq_linear(k: int, n: int, group_size: int = 128, seed: int = 0):
    """Integer weights/zeros, fp16 scales, and the packed tensors (via AutoAWQ's packer)."""
    g = torch.Generator().manual_seed(seed)
    iw = torch.randint(0, 16, (k, n), generator=g, dtype=torch.int32)
    iz = torch.randint(0, 16, (k // group_size, n), generator=g, dtype=torch.int32)
    scales = (torch.rand(k // group_size, n, generator=g) * 0.02 + 0.002).half()
    return iw, iz, autoawq_pack(iw), autoawq_pack(iz), scales


def _hf_from_state_dict(hf_config_dict: dict, sd: dict):
    """Build an fp16 HF ``Qwen2ForCausalLM`` directly from a state dict (no random init)."""
    import transformers
    # Import the modeling module BEFORE entering the meta-device context: transformers
    # imports it lazily and (in 4.51) runs tensor code at import time, which fails on meta.
    from transformers.models.qwen2 import modeling_qwen2

    cfg = transformers.Qwen2Config(**hf_config_dict)
    cfg.torch_dtype = torch.float16
    with torch.device("meta"):
        model = modeling_qwen2.Qwen2ForCausalLM(cfg)
    sd = dict(sd)
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    model.load_state_dict(sd, strict=True, assign=True)
    # Non-persistent buffers (rotary inv_freq) were created on the meta device.
    model.model.rotary_emb = modeling_qwen2.Qwen2RotaryEmbedding(config=cfg)
    return model.eval()


def build_hf_reference(model_dir, hf_config_dict, device="cpu"):
    """HF ``Qwen2ForCausalLM`` (fp16) whose Linear weights are the AutoAWQ-dequantized
    checkpoint weights. Materializes full fp16 weights in host RAM - fine for a
    test reference, never used by the engine.
    """
    from pathlib import Path

    from safetensors import safe_open

    sd = {}
    with safe_open(str(Path(model_dir) / "model.safetensors"), framework="pt", device="cpu") as f:
        for name in f.keys():
            if name.endswith(".qweight"):
                base = name[: -len(".qweight")]
                w = autoawq_dequantize_gemm(f.get_tensor(name), f.get_tensor(base + ".qzeros"),
                                            f.get_tensor(base + ".scales"), 4, 128)
                sd[base + ".weight"] = w.t().contiguous()  # [in, out] -> [out, in]
            elif not name.endswith((".qzeros", ".scales")):
                sd[name] = f.get_tensor(name)  # biases, norms, embeddings, lm_head
    return _hf_from_state_dict(hf_config_dict, sd).to(device)


def build_hf_from_engine_model(model):
    """HF reference for an in-memory (synthetic) engine model, using the same packed
    weights dequantized by AutoAWQ's own routine."""
    cfg = model.cfg
    sd = {"model.embed_tokens.weight": model.embed.cpu(), "model.norm.weight": model.final_norm.cpu()}
    names = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj",
             "o": "self_attn.o_proj", "gate": "mlp.gate_proj", "up": "mlp.up_proj",
             "down": "mlp.down_proj"}
    for i, lw in enumerate(model.layers):
        pre = f"model.layers.{i}."
        for short, hf_name in names.items():
            lin = getattr(lw, short)
            w = autoawq_dequantize_gemm(lin.qweight.cpu(), lin.qzeros.cpu(), lin.scales.cpu(), 4,
                                        cfg.group_size)
            sd[pre + hf_name + ".weight"] = w.t().contiguous()
            if lin.bias is not None:
                sd[pre + hf_name + ".bias"] = lin.bias.cpu()
        sd[pre + "input_layernorm.weight"] = lw.input_ln.cpu()
        sd[pre + "post_attention_layernorm.weight"] = lw.post_ln.cpu()
    hf_cfg = dict(
        vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size, intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_layers, num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_kv_heads, rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta, max_position_embeddings=4096, tie_word_embeddings=True,
        use_sliding_window=False, attention_dropout=0.0,
    )
    return _hf_from_state_dict(hf_cfg, sd)
