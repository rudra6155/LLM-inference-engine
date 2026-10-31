#!/usr/bin/env python3
"""Mutation check of the test suite: inject one plausible bug at a time into a *copy* of
the source and confirm the named tests fail.

A surviving mutant means the tests would not catch that bug. Writes
results/mutation_check.txt. Takes several minutes (the engine tests run Triton in the
CPU interpreter when no GPU is present).

    python scripts/mutation_check.py [--only SUBSTRING]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (name, file, old text, new text, pytest args)
MUTANTS = [
    ("gemm: wrong nibble order", "src/engine/kernels/dequant_gemm.py",
     "shifts = ((j % 2) * 4 + j // 2) * 4", "shifts = j * 4",
     ["tests/test_kernels.py", "-k", "awq_gemm_matches_reference"]),
    ("gemm: split-K drops the last partial", "src/engine/kernels/dequant_gemm.py",
     "    for s in range(0, SPLIT_K):\n        acc += tl.load", "    for s in range(0, SPLIT_K - 1):\n        acc += tl.load",
     ["tests/test_kernels.py", "-k", "split_k"]),
    ("attention: wrong GQA head mapping", "src/engine/kernels/fused_attention.py",
     "q_head = kv_head * GROUP + offs_g", "q_head = kv_head + offs_g * (G_PAD // GROUP)",
     ["tests/test_kernels.py", "-k", "decode_attention_matches_reference"]),
    ("attention: positions beyond seq_len not masked", "src/engine/kernels/fused_attention.py",
     's = tl.where(mask_n[None, :], s, float("-inf"))', "s = s",
     ["tests/test_kernels.py", "-k", "decode_attention_matches_reference"]),
    ("kv cache: write at the wrong offset", "src/engine/generate.py",
     "self.data[layer, 0, :, pos:pos + t] = k.transpose(0, 1)",
     "self.data[layer, 0, :, pos + 1:pos + t + 1] = k.transpose(0, 1)",
     ["tests/test_engine.py", "-k", "kv_cache_write"]),
    ("model: q/k/v bias dropped", "src/engine/model.py",
     "            return awq_gemm(x, self.qweight, self.qzeros, self.scales, self.bias,",
     "            return awq_gemm(x, self.qweight, self.qzeros, self.scales, None,",
     ["tests/test_engine.py", "-k", "first_token_matches_reference"]),
    ("model: RoPE applied to q at the wrong position", "src/engine/model.py",
     "q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)",
     "q, k = apply_rope(q, self.cos[pos0 + 1:pos0 + t + 1], self.sin[pos0 + 1:pos0 + t + 1]), apply_rope(k, cos, sin)",
     ["tests/test_engine.py", "-k", "huggingface"]),
    ("model: k not rotated", "src/engine/model.py",
     "q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)", "q, k = apply_rope(q, cos, sin), k",
     ["tests/test_engine.py", "-k", "huggingface"]),
    ("model: wrong RoPE theta", "src/engine/model.py",
     "inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))",
     "inv_freq = 1.0 / ((theta * 2) ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))",
     ["tests/test_engine.py", "-k", "huggingface"]),
    ("model: gate/up swapped in SwiGLU", "src/engine/model.py",
     "F.silu(lw.gate(x, gemm)) * lw.up(x, gemm)", "F.silu(lw.up(x, gemm)) * lw.gate(x, gemm)",
     ["tests/test_engine.py", "-k", "huggingface"]),
    ("model: GQA heads repeated in the wrong order", "src/engine/model.py",
     "k = cache.k(li)[:, :s].repeat_interleave(cfg.gqa_group, dim=0)",
     "k = cache.k(li)[:, :s].repeat(cfg.gqa_group, 1, 1)",
     ["tests/test_engine.py", "-k", "huggingface"]),
]


def run_mutant(name, rel, old, new, pytest_args) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for d in ("src", "tests"):
            shutil.copytree(ROOT / d, work / d, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(ROOT / "pyproject.toml", work / "pyproject.toml")
        target = work / rel
        text = target.read_text()
        if text.count(old) != 1:
            return False, f"pattern not found exactly once in {rel} (found {text.count(old)})"
        target.write_text(text.replace(old, new))
        r = subprocess.run([sys.executable, "-m", "pytest", *pytest_args, "-q", "-x", "-p", "no:cacheprovider"],
                           cwd=work, capture_output=True, text=True)
        tail = (r.stdout.strip().splitlines() or [""])[-1]
        return r.returncode != 0, tail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    lines, survived = [], 0
    for m in MUTANTS:
        if args.only not in m[0]:
            continue
        caught, tail = run_mutant(*m)
        survived += not caught
        line = f"{'CAUGHT  ' if caught else 'SURVIVED'}  {m[0]:50s} [{tail}]"
        print(line, flush=True)
        lines.append(line)
    lines.append(f"{len(lines) - survived} of {len(lines)} mutants caught")
    print(lines[-1])
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "mutation_check.txt").write_text("\n".join(lines) + "\n")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
