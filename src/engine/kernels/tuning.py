"""Tuned kernel launch configurations, measured on the target GPU.

``scripts/tune_kernels.py`` sweeps tile shapes / ``num_warps`` / ``num_stages`` / split-K
for the decode GEMM shapes and tile / split-KV options for decode attention, checks every
candidate against a reference, times it on the GPU, and writes a JSON file. This module
reads that file back (``EngineOptions(tuned_config=path)``) and validates it, so a stale
or hand-edited file fails loudly instead of mis-launching a kernel.

Nothing here is a guess: if no file is given the engine uses ``default_config`` and the
fused attention defaults, exactly as before.

File layout (``SCHEMA`` 1)::

    {"schema": 1,
     "meta": {...device, versions, timing method, how each pick was made...},
     "awq_gemm": {"2048x11008": {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 2,
                                  "num_warps": 4, "num_stages": 3}, ...},      # key "KxN", used when M == 1
     "attention": {"block_n": 64, "num_warps": 4, "num_stages": 3, "num_splits": 1}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .dequant_gemm import validate_config

SCHEMA = 1
ATTENTION_KEYS = {"block_n", "num_warps", "num_stages", "num_splits"}


def gemm_key(k: int, n: int) -> str:
    return f"{k}x{n}"


def validate_attention(cfg: dict) -> dict:
    unknown = set(cfg) - ATTENTION_KEYS
    if unknown:
        raise ValueError(f"bad attention config {cfg}: unknown keys {sorted(unknown)}")
    for key, value in cfg.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"bad attention config {cfg}: {key} must be a positive int")
    block_n = cfg.get("block_n", 64)
    if block_n & (block_n - 1) or block_n < 16:
        raise ValueError(f"bad attention config {cfg}: block_n must be a power of two >= 16")
    return dict(cfg)


@dataclass(frozen=True)
class TunedConfigs:
    gemm: dict[tuple[int, int], dict] = field(default_factory=dict)  # (K, N) -> awq_gemm config
    attention: dict = field(default_factory=dict)                    # kwargs of fused_decode_attention
    meta: dict = field(default_factory=dict)
    path: str | None = None

    def gemm_config(self, m: int, k: int, n: int) -> dict | None:
        """Tuned config for a decode-shaped GEMM (M == 1); ``None`` means use the default."""
        return self.gemm.get((k, n)) if m == 1 else None

    def attention_kwargs(self) -> dict:
        return dict(self.attention)

    def describe(self) -> str:
        return f"{self.path or '<memory>'}: {len(self.gemm)} gemm shapes, attention={self.attention or 'default'}"

    # -- (de)serialization ----------------------------------------------- #
    @classmethod
    def from_dict(cls, d: dict, *, group_size: int = 128, path: str | None = None) -> "TunedConfigs":
        if d.get("schema") != SCHEMA:
            raise ValueError(f"tuned config schema {d.get('schema')!r} != {SCHEMA}")
        gemm: dict[tuple[int, int], dict] = {}
        for key, cfg in d.get("awq_gemm", {}).items():
            try:
                k_str, n_str = key.split("x")
                k, n = int(k_str), int(n_str)
            except ValueError as e:
                raise ValueError(f"bad awq_gemm key {key!r} (expected 'KxN')") from e
            gemm[(k, n)] = validate_config(cfg, k, group_size)
        return cls(gemm, validate_attention(d.get("attention", {})), dict(d.get("meta", {})), path)

    @classmethod
    def load(cls, path: str | Path, *, group_size: int = 128) -> "TunedConfigs":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"tuned config {p} not found (run scripts/tune_kernels.py on the GPU first)")
        return cls.from_dict(json.loads(p.read_text()), group_size=group_size, path=str(p))

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "meta": self.meta,
                "awq_gemm": {gemm_key(k, n): cfg for (k, n), cfg in sorted(self.gemm.items())},
                "attention": self.attention}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")
