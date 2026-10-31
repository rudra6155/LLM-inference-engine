#!/usr/bin/env python3
"""KV-cache memory model from first principles, for the engine's model.

    bytes per token = 2 (K and V) * num_layers * num_kv_heads * head_dim * dtype_bytes

Also prints what that means for an 8 GB card (weights are the tensors in the checkpoint
with the tied lm_head stored once). Everything is computed from the model config; run it
to reproduce the numbers quoted in FINDINGS.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from engine.model import QWEN2_5_3B as C  # noqa: E402
from engine.model import dtype_bytes, kv_cache_bytes_per_token  # noqa: E402

# Bytes of every tensor in the checkpoint, read from its safetensors header (2026-10-29),
# minus the duplicate lm_head (byte-identical to embed_tokens, verified by sha256).
CHECKPOINT_BYTES = 2_686_599_168
LM_HEAD_BYTES = 622_329_856


def main() -> int:
    b = dtype_bytes(torch.float16)
    per_tok = kv_cache_bytes_per_token(C, torch.float16)
    print("Qwen2.5-3B-Instruct-AWQ")
    print(f"  num_layers={C.num_layers}  num_kv_heads={C.num_kv_heads}  head_dim={C.head_dim} "
          f"(= hidden {C.hidden_size} / heads {C.num_heads})  dtype=fp16 ({b} B)")
    print(f"  bytes/token = 2 * {C.num_layers} * {C.num_kv_heads} * {C.head_dim} * {b} = {per_tok:,} B "
          f"({per_tok / 1024:.0f} KiB)")
    print(f"  (MHA would need {C.num_heads // C.num_kv_heads}x more: "
          f"{per_tok * C.num_heads // C.num_kv_heads:,} B/token)\n")
    print(f"{'context':>8} {'KV bytes':>16} {'KV MiB':>10}")
    for n in (128, 512, 1024, 2048, 4096, 8192, 32768):
        print(f"{n:>8} {per_tok * n:>16,} {per_tok * n / 2**20:>10.1f}")
    weights = CHECKPOINT_BYTES - LM_HEAD_BYTES
    print(f"\nweights held once (tied lm_head): {weights:,} B = {weights / 2**30:.2f} GiB")
    for n in (1024, 4096, 32768):
        total = weights + per_tok * n
        print(f"weights + KV at {n:>5} tokens: {total / 2**30:.2f} GiB  ({total / (8 * 2**30) * 100:.0f}% of 8 GiB, "
              f"before CUDA context / activations / allocator slack)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
