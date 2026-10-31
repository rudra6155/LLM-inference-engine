"""Manual KV cache and the autoregressive (greedy) generation loop.

This is the engine's primary generation path. It never calls
``transformers``' ``generate()``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

import torch

from .graph import DeviceDecoder
from .model import DevicePos, ModelConfig, QwenEngineModel, dtype_bytes, kv_cache_bytes_per_token


class KVCache:
    """Preallocated KV cache: one tensor ``[layers, 2, kv_heads, max_seq, head_dim]``.

    Its size is exactly ``kv_cache_bytes_per_token(cfg, dtype) * max_seq_len``.
    """

    def __init__(self, cfg: ModelConfig, max_seq_len: int, device, dtype: torch.dtype = torch.float16):
        self.cfg = cfg
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.data = torch.zeros(
            (cfg.num_layers, 2, cfg.num_kv_heads, max_seq_len, cfg.head_dim),
            dtype=dtype, device=device,
        )
        self.length = 0

    @property
    def nbytes(self) -> int:
        return self.data.numel() * dtype_bytes(self.dtype)

    @property
    def bytes_per_token(self) -> int:
        return kv_cache_bytes_per_token(self.cfg, self.dtype)

    def k(self, layer: int) -> torch.Tensor:
        return self.data[layer, 0]  # [kv_heads, max_seq, head_dim]

    def v(self, layer: int) -> torch.Tensor:
        return self.data[layer, 1]

    def write(self, layer: int, pos, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store ``k`` / ``v`` of shape ``[T, kv_heads, head_dim]`` at positions ``pos..pos+T``.

        ``pos`` is a Python int, or a ``DevicePos`` (T == 1): then the slot is chosen by an
        ``index_copy_`` from device memory, which is what lets a CUDA graph replay at any position
        (Hugging Face's static cache writes the same way, for the same reason).
        """
        if isinstance(pos, DevicePos):
            self.data[layer, 0].index_copy_(1, pos.pos, k.transpose(0, 1))
            self.data[layer, 1].index_copy_(1, pos.pos, v.transpose(0, 1))
            return
        t = k.shape[0]
        if pos + t > self.max_seq_len:
            raise ValueError("KV cache overflow")
        self.data[layer, 0, :, pos:pos + t] = k.transpose(0, 1)
        self.data[layer, 1, :, pos:pos + t] = v.transpose(0, 1)

    def reset(self) -> None:
        self.length = 0  # stale entries are never read; positions are tracked explicitly


@dataclass
class GenerationResult:
    token_ids: list[int]
    prompt_tokens: int
    ttft_s: float
    itl_s: list[float] = field(default_factory=list)  # one entry per token after the first


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def generate(
    model: QwenEngineModel,
    prompt_ids: Iterable[int],
    max_new_tokens: int,
    *,
    cache: KVCache | None = None,
    eos_token_ids: Iterable[int] = (),
    ignore_eos: bool = False,
    decoder: DeviceDecoder | None = None,
) -> GenerationResult:
    """Greedy decoding. Each token is copied to the host as it is produced, so the
    recorded times are what a caller would observe (they include Python and
    kernel-launch overhead).

    ``decoder``: a ``DeviceDecoder`` (``engine.graph``) bound to ``cache``. If it is usable the decode
    loop runs through it - replaying a captured CUDA graph, or the same device-position step eagerly -
    otherwise (not usable, e.g. capture failed) the ordinary eager loop runs and the caller can read
    ``decoder.describe()`` for the reason. Prefill is always eager. Any capture happens before the clock starts.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    device = model.device
    ids = torch.as_tensor(list(prompt_ids), dtype=torch.long, device=device)
    if ids.numel() == 0:
        raise ValueError("empty prompt")
    if ids.numel() + max_new_tokens > model.max_seq_len:
        raise ValueError("prompt + max_new_tokens exceeds the model's max_seq_len")
    if decoder is not None:
        if cache is not None and cache is not decoder.cache:
            raise ValueError("the decoder is bound to its own KV cache; pass that cache or none")
        cache = decoder.cache
        decoder.ensure_ready()  # warm-up + capture, outside the timed region
    if cache is None:
        cache = KVCache(model.cfg, model.max_seq_len, device)
    cache.reset()
    eos = set(eos_token_ids)

    _sync(device)
    t_start = time.perf_counter()
    tok = torch.argmax(model.prefill(ids, cache))
    token = int(tok.item())
    pos = int(ids.numel())
    use_decoder = decoder is not None and decoder.usable
    if use_decoder:
        decoder.start(tok, pos)  # counted in TTFT: it is work the engine does to begin decoding
    t_prev = time.perf_counter()
    result = GenerationResult([token], int(ids.numel()), t_prev - t_start)

    while len(result.token_ids) < max_new_tokens:
        if not ignore_eos and token in eos:
            break
        if use_decoder:
            token = decoder.step()
        else:
            tok = torch.argmax(model.decode_step(tok, cache, pos))
            token = int(tok.item())
        now = time.perf_counter()
        result.itl_s.append(now - t_prev)
        t_prev = now
        result.token_ids.append(token)
        pos += 1
    cache.length = pos
    return result
