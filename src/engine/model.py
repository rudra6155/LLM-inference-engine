"""Quantized Qwen2 model for batch-size-1 inference.

Loading: ``load_awq_model(..., loader="autoawq")`` loads the 4-bit AWQ checkpoint
with ``autoawq`` (default) and manages the packed
``qweight / qzeros / scales`` tensors. ``loader="safetensors"`` reads the same
tensors directly from the checkpoint file; it exists for CPU-only tests and must
be requested explicitly.

Execution: the default path uses the Triton fused dequantize+GEMM, the Triton
fused decode attention and the C++ RMSNorm. ``EngineOptions`` can swap each for a
PyTorch reference; the ``"reference"`` GEMM dequantizes one weight matrix at a
time and exists ONLY for tests and profiling baselines - the engine never uses it
by default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F

from .cpp_ops import rmsnorm as cpp_rmsnorm
from .cpp_ops import rmsnorm_reference
from .kernels.dequant_gemm import awq_gemm, dequantize_reference, pack_awq
from .kernels.fused_attention import fused_decode_attention
from .kernels.tuning import TunedConfigs

# --------------------------------------------------------------------------- #
# Configuration and the KV-cache memory model                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    group_size: int = 128
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        # Qwen2's config.json has no head_dim key; HF derives it this way.
        return self.hidden_size // self.num_heads

    @property
    def gqa_group(self) -> int:
        return self.num_heads // self.num_kv_heads

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        c = json.loads(Path(path).read_text())
        q = c.get("quantization_config", {})
        if q:
            if q.get("quant_method") != "awq" or q.get("bits") != 4 or not q.get("zero_point"):
                raise ValueError(f"unsupported quantization_config: {q}")
            if q.get("version", "gemm").lower() != "gemm":
                raise ValueError(f"only AWQ 'gemm' layout is supported, got {q.get('version')}")
        return cls(
            hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            num_layers=c["num_hidden_layers"],
            num_heads=c["num_attention_heads"],
            num_kv_heads=c["num_key_value_heads"],
            vocab_size=c["vocab_size"],
            rms_norm_eps=c["rms_norm_eps"],
            rope_theta=c["rope_theta"],
            group_size=q.get("group_size", 128),
            tie_word_embeddings=c.get("tie_word_embeddings", True),
        )


# Values read from Qwen/Qwen2.5-3B-Instruct-AWQ config.json (2026-10-29).
QWEN2_5_3B = ModelConfig(
    hidden_size=2048,
    intermediate_size=11008,
    num_layers=36,
    num_heads=16,
    num_kv_heads=2,
    vocab_size=151936,
    rms_norm_eps=1e-6,
    rope_theta=1_000_000.0,
)


def dtype_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def kv_cache_bytes_per_token(cfg: ModelConfig, dtype: torch.dtype = torch.float16) -> int:
    """First-principles KV size: 2 (K and V) * layers * kv_heads * head_dim * bytes."""
    return 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * dtype_bytes(dtype)


# --------------------------------------------------------------------------- #
# Weights                                                                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EngineOptions:
    gemm: Literal["triton", "reference"] = "triton"
    attention: Literal["triton", "reference"] = "triton"
    rmsnorm: Literal["cpp", "reference"] = "cpp"
    # Path of a file written by scripts/tune_kernels.py. ``None`` keeps the untuned defaults.
    tuned_config: str | None = None


@dataclass(frozen=True)
class DevicePos:
    """The decode position held in device memory, so one CUDA graph can serve every position.

    A graph freezes its kernel arguments, so a Python ``pos`` would be replayed forever. With
    these tensors the position is *data*: the KV write uses ``index_copy_``, RoPE uses
    ``index_select`` and attention reads its length from ``seqlen``.
    """

    pos: torch.Tensor      # int64 [1]: index of the token being processed
    seqlen: torch.Tensor   # int32 [1]: pos + 1, the number of cached tokens attention reads

    @classmethod
    def from_tensor(cls, pos: torch.Tensor) -> "DevicePos":
        """``pos``: int64 ``[1]``. ``seqlen`` is derived on the device (no host round trip)."""
        return cls(pos, (pos + 1).to(torch.int32))


class QuantLinear:
    """A 4-bit AWQ linear layer. Holds only the packed tensors."""

    def __init__(self, qweight, qzeros, scales, bias, group_size: int):
        self.qweight, self.qzeros, self.scales, self.bias = qweight, qzeros, scales, bias
        self.group_size = group_size
        self.in_features = qweight.shape[0]
        self.out_features = qweight.shape[1] * 8

    def __call__(self, x: torch.Tensor, gemm: str, tuned: TunedConfigs | None = None) -> torch.Tensor:
        if gemm == "triton":
            cfg = tuned.gemm_config(x.reshape(-1, x.shape[-1]).shape[0], self.in_features, self.out_features) if tuned else None
            return awq_gemm(x, self.qweight, self.qzeros, self.scales, self.bias,
                            group_size=self.group_size, config=cfg)
        # REFERENCE ONLY (tests / profiling baseline): materializes this one matrix in fp16.
        w = dequantize_reference(self.qweight, self.qzeros, self.scales, self.group_size)
        y = x.float() @ w.float()
        if self.bias is not None:
            y = y + self.bias.float()
        return y.to(x.dtype)

    def weight_bytes(self) -> int:
        n = self.qweight.nbytes + self.qzeros.nbytes + self.scales.nbytes
        return n + (self.bias.nbytes if self.bias is not None else 0)


@dataclass
class LayerWeights:
    q: QuantLinear
    k: QuantLinear
    v: QuantLinear
    o: QuantLinear
    gate: QuantLinear
    up: QuantLinear
    down: QuantLinear
    input_ln: torch.Tensor
    post_ln: torch.Tensor


_PROJ = {
    "q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj",
    "o": "self_attn.o_proj", "gate": "mlp.gate_proj", "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}


# --------------------------------------------------------------------------- #
# RoPE                                                                        #
# --------------------------------------------------------------------------- #


def rope_tables(head_dim: int, theta: float, max_seq_len: int, device, dtype=torch.float16):
    """cos/sin tables computed in fp32 then cast, as HF's Qwen2RotaryEmbedding does."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(device=device, dtype=dtype), emb.sin().to(device=device, dtype=dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x``: ``[T, H, D]``; ``cos`` / ``sin``: ``[T, D]``."""
    return x * cos[:, None, :] + _rotate_half(x) * sin[:, None, :]


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #


class QwenEngineModel:
    """Qwen2 forward pass over packed 4-bit weights, batch size 1."""

    def __init__(self, cfg: ModelConfig, embed: torch.Tensor, layers: list[LayerWeights],
                 final_norm: torch.Tensor, options: EngineOptions | None = None,
                 max_seq_len: int = 2048):
        self.cfg = cfg
        self.embed = embed  # [vocab, hidden], also the (tied) lm_head
        self.layers = layers
        self.final_norm = final_norm
        self.options = options or EngineOptions()
        self.device = embed.device
        self.max_seq_len = max_seq_len
        self.cos, self.sin = rope_tables(cfg.head_dim, cfg.rope_theta, max_seq_len, self.device)
        self.tuned = (TunedConfigs.load(self.options.tuned_config, group_size=cfg.group_size)
                      if self.options.tuned_config else None)
        self._attn_kwargs = self.tuned.attention_kwargs() if self.tuned else {}

    def with_options(self, options: EngineOptions) -> "QwenEngineModel":
        """Same weights (no copy), different execution options."""
        return QwenEngineModel(self.cfg, self.embed, self.layers, self.final_norm, options,
                               self.max_seq_len)

    # -- weight accounting ------------------------------------------------- #
    def weight_bytes(self) -> dict[str, int]:
        quant = sum(
            getattr(l, p).weight_bytes() for l in self.layers for p in _PROJ
        )
        norms = sum(l.input_ln.nbytes + l.post_ln.nbytes for l in self.layers) + self.final_norm.nbytes
        return {"quantized_linears": quant, "embedding_and_lm_head": self.embed.nbytes, "norms": norms}

    # -- building blocks --------------------------------------------------- #
    def _norm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.options.rmsnorm == "cpp":
            return cpp_rmsnorm(x, w, self.cfg.rms_norm_eps)
        return rmsnorm_reference(x, w, self.cfg.rms_norm_eps)

    def _attention(self, li: int, q: torch.Tensor, cache, pos0, t: int) -> torch.Tensor:
        cfg = self.cfg
        if isinstance(pos0, DevicePos):
            if self.options.attention != "triton":
                raise ValueError("device-position decode needs attention='triton' (the reference path slices by a Python int)")
            out = fused_decode_attention(q[0], cache.k(li), cache.v(li), pos0.seqlen, **self._attn_kwargs)
            return out.reshape(1, cfg.num_heads * cfg.head_dim)
        if t == 1 and self.options.attention == "triton":
            out = fused_decode_attention(q[0], cache.k(li), cache.v(li), pos0 + 1, **self._attn_kwargs)  # [Hq, D]
            return out.reshape(1, cfg.num_heads * cfg.head_dim)
        s = pos0 + t
        k = cache.k(li)[:, :s].repeat_interleave(cfg.gqa_group, dim=0)  # [Hq, S, D]
        v = cache.v(li)[:, :s].repeat_interleave(cfg.gqa_group, dim=0)
        qh = q.transpose(0, 1)  # [Hq, T, D]
        o = F.scaled_dot_product_attention(qh[None], k[None], v[None], is_causal=(t > 1))
        return o[0].transpose(0, 1).reshape(t, cfg.num_heads * cfg.head_dim)

    def _forward_layers(self, h: torch.Tensor, cache, pos0) -> torch.Tensor:
        """``pos0`` is the position of the first token: a Python int, or a ``DevicePos`` (T == 1 only)."""
        cfg, gemm, tuned = self.cfg, self.options.gemm, self.tuned
        t = h.shape[0]
        if isinstance(pos0, DevicePos):
            if t != 1:
                raise ValueError("a device-resident position supports a single token (decode) only")
            cos, sin = self.cos.index_select(0, pos0.pos), self.sin.index_select(0, pos0.pos)  # [1, D]
        else:
            cos, sin = self.cos[pos0:pos0 + t], self.sin[pos0:pos0 + t]
        for li, lw in enumerate(self.layers):
            x = self._norm(h, lw.input_ln)
            q = lw.q(x, gemm, tuned).view(t, cfg.num_heads, cfg.head_dim)
            k = lw.k(x, gemm, tuned).view(t, cfg.num_kv_heads, cfg.head_dim)
            v = lw.v(x, gemm, tuned).view(t, cfg.num_kv_heads, cfg.head_dim)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            cache.write(li, pos0, k, v)
            h = h + lw.o(self._attention(li, q, cache, pos0, t), gemm, tuned)
            x = self._norm(h, lw.post_ln)
            h = h + lw.down(F.silu(lw.gate(x, gemm, tuned)) * lw.up(x, gemm, tuned), gemm, tuned)
        return h

    def _logits(self, h_last: torch.Tensor) -> torch.Tensor:
        h = self._norm(h_last, self.final_norm)
        return F.linear(h, self.embed)[0]  # [vocab]

    # -- public API -------------------------------------------------------- #
    def prefill(self, input_ids: torch.Tensor, cache) -> torch.Tensor:
        """Process the whole prompt into an empty cache; return last-token logits ``[vocab]``."""
        t = input_ids.shape[0]
        if t > cache.max_seq_len:
            raise ValueError(f"prompt of {t} tokens exceeds cache capacity {cache.max_seq_len}")
        h = F.embedding(input_ids, self.embed)
        h = self._forward_layers(h, cache, 0)
        return self._logits(h[-1:])

    def decode_step(self, token: torch.Tensor, cache, pos: int) -> torch.Tensor:
        """Process one token at position ``pos``; return logits ``[vocab]``."""
        if pos >= cache.max_seq_len:
            raise ValueError("KV cache is full")
        h = F.embedding(token.reshape(1), self.embed)
        h = self._forward_layers(h, cache, pos)
        return self._logits(h)

    def decode_step_device(self, token: torch.Tensor, cache, dpos: DevicePos) -> torch.Tensor:
        """Same as ``decode_step`` but the position lives in device memory (``dpos``), so every
        argument of every kernel launched is either a tensor or a constant: the step can be captured
        in a CUDA graph and replayed at any position. No host synchronization happens here, so the
        caller must keep ``0 <= dpos.pos < cache.max_seq_len`` (a Python-side check would need a
        device-to-host copy)."""
        h = F.embedding(token.reshape(1), self.embed)
        h = self._forward_layers(h, cache, dpos)
        return self._logits(h)


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #

_DOWNLOAD_PATTERNS = ["config.json", "generation_config.json", "model*.safetensors",
                      "model.safetensors.index.json", "tokenizer*", "vocab.json", "merges.txt"]


def resolve_model_dir(model: str) -> Path:
    """Return a local directory for ``model`` (local path or Hugging Face repo id)."""
    p = Path(model)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, allow_patterns=_DOWNLOAD_PATTERNS))


def _assemble(cfg: ModelConfig, get, options, max_seq_len, embed) -> QwenEngineModel:
    layers = []
    for i in range(cfg.num_layers):
        pre = f"model.layers.{i}."

        def lin(name: str) -> QuantLinear:
            base = pre + _PROJ[name]
            return QuantLinear(get(base + ".qweight"), get(base + ".qzeros"),
                               get(base + ".scales"), get(base + ".bias", optional=True),
                               cfg.group_size)

        layers.append(LayerWeights(
            q=lin("q"), k=lin("k"), v=lin("v"), o=lin("o"),
            gate=lin("gate"), up=lin("up"), down=lin("down"),
            input_ln=get(pre + "input_layernorm.weight"),
            post_ln=get(pre + "post_attention_layernorm.weight"),
        ))
    return QwenEngineModel(cfg, embed, layers, get("model.norm.weight"), options, max_seq_len)


def _load_safetensors(model_dir: Path, cfg: ModelConfig, device, options, max_seq_len):
    from safetensors import safe_open

    path = model_dir / "model.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (sharded checkpoints are not supported)")
    # Tensors are read on the CPU and moved one at a time: this does not depend on
    # safetensors' direct-to-CUDA path, and host memory never holds more than one tensor
    # beyond the 2 GB checkpoint mapping.
    with safe_open(str(path), framework="pt", device="cpu") as f:
        def get(name, optional=False):
            if name not in f.keys():
                if optional:
                    return None
                raise KeyError(name)
            return f.get_tensor(name).to(device)

        embed_cpu = f.get_tensor("model.embed_tokens.weight")
        if cfg.tie_word_embeddings and "lm_head.weight" in f.keys():
            # The config says tied; make sure tying loses nothing.
            if not torch.equal(f.get_tensor("lm_head.weight"), embed_cpu):
                raise ValueError("lm_head.weight differs from embed_tokens.weight but config ties them")
        return _assemble(cfg, get, options, max_seq_len, embed_cpu.to(device))


def _load_autoawq(model_dir: Path, cfg: ModelConfig, device, options, max_seq_len):
    # Imported lazily: autoawq is deprecated and only needed on this path.
    from awq import AutoAWQForCausalLM

    dev = torch.device(device)
    if dev.type != "cuda":
        raise RuntimeError("the autoawq loader requires a CUDA device; use loader='safetensors' on CPU")
    awq_model = AutoAWQForCausalLM.from_quantized(
        str(model_dir), fuse_layers=False, safetensors=True, max_seq_len=max_seq_len,
        device_map={"": dev.index or 0},
    )
    hf = awq_model.model
    tensors: dict[str, torch.Tensor] = {}
    for i, layer in enumerate(hf.model.layers):
        pre = f"model.layers.{i}."
        for short, attr in _PROJ.items():
            mod = layer
            for part in attr.split("."):
                mod = getattr(mod, part)
            tensors[pre + attr + ".qweight"] = mod.qweight
            tensors[pre + attr + ".qzeros"] = mod.qzeros
            tensors[pre + attr + ".scales"] = mod.scales
            if mod.bias is not None:
                tensors[pre + attr + ".bias"] = mod.bias
        tensors[pre + "input_layernorm.weight"] = layer.input_layernorm.weight.data
        tensors[pre + "post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight.data
    tensors["model.norm.weight"] = hf.model.norm.weight.data
    embed = hf.model.embed_tokens.weight.data
    if cfg.tie_word_embeddings and hf.lm_head.weight.data_ptr() != embed.data_ptr():
        if not torch.equal(hf.lm_head.weight.data, embed):
            raise ValueError("lm_head.weight differs from embed_tokens.weight but config ties them")

    def get(name, optional=False):
        if name not in tensors:
            if optional:
                return None
            raise KeyError(name)
        return tensors[name]

    model = _assemble(cfg, get, options, max_seq_len, embed)
    del awq_model, hf  # the engine keeps its own references to the tensors it uses
    return model


def load_awq_model(model: str, *, loader: Literal["autoawq", "safetensors"] = "autoawq",
                   device: str | torch.device = "cuda", max_seq_len: int = 2048,
                   options: EngineOptions | None = None) -> QwenEngineModel:
    if loader not in ("autoawq", "safetensors"):
        raise ValueError(f"unknown loader {loader!r} (choose 'autoawq' or 'safetensors'; there is no implicit default)")
    model_dir = resolve_model_dir(model)
    cfg = ModelConfig.from_json(model_dir / "config.json")
    if loader == "autoawq":
        return _load_autoawq(model_dir, cfg, device, options, max_seq_len)
    return _load_safetensors(model_dir, cfg, device, options, max_seq_len)


def build_synthetic_model(cfg: ModelConfig, *, seed: int = 0, device="cpu",
                          options: EngineOptions | None = None,
                          max_seq_len: int = 256, embed_std: float = 0.05) -> QwenEngineModel:
    """Random 4-bit weights in the real AWQ layout, for tests that must run anywhere.

    ``embed_std`` is small on purpose: with large embeddings the residual stream is
    dominated by the input token, the model just echoes it, and an end-to-end
    comparison would pass even if attention were broken.
    """
    g = torch.Generator().manual_seed(seed)
    dev = torch.device(device)

    def rand_lin(k: int, n: int, bias: bool) -> QuantLinear:
        iw = torch.randint(0, 16, (k, n), generator=g)
        iz = torch.randint(6, 10, (k // cfg.group_size, n), generator=g)
        qw, qz = pack_awq(iw, iz, cfg.group_size)
        scales = (torch.rand(k // cfg.group_size, n, generator=g) * 0.01 + 0.002).half()
        b = (torch.randn(n, generator=g) * 0.1).half().to(dev) if bias else None
        return QuantLinear(qw.to(dev), qz.to(dev), scales.to(dev), b, cfg.group_size)

    h, i = cfg.hidden_size, cfg.intermediate_size
    kv = cfg.num_kv_heads * cfg.head_dim
    layers = []
    for _ in range(cfg.num_layers):
        layers.append(LayerWeights(
            q=rand_lin(h, h, True), k=rand_lin(h, kv, True), v=rand_lin(h, kv, True),
            o=rand_lin(h, h, False), gate=rand_lin(h, i, False), up=rand_lin(h, i, False),
            down=rand_lin(i, h, False),
            input_ln=(1 + 0.1 * torch.randn(h, generator=g)).half().to(dev),
            post_ln=(1 + 0.1 * torch.randn(h, generator=g)).half().to(dev),
        ))
    embed = (torch.randn(cfg.vocab_size, h, generator=g) * embed_std).half().to(dev)
    final_norm = (1 + 0.1 * torch.randn(h, generator=g)).half().to(dev)
    return QwenEngineModel(cfg, embed, layers, final_norm, options, max_seq_len)
