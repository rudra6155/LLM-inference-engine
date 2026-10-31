#!/usr/bin/env python3
"""Log the maximum |kernel - reference| for the Triton kernels and the C++ op.

Writes results/kernel_max_diff.csv. The `mode` column says how the Triton kernels ran:
`interpreter` (TRITON_INTERPRET=1 on CPU: validates logic and numerics only) or
`compiled-cuda` (real GPU code). Run this on the GPU machine to get the compiled rows.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

from engine.cpp_ops import rmsnorm, rmsnorm_reference  # noqa: E402
from engine.kernels.dequant_gemm import awq_gemm, awq_gemm_reference, pack_awq  # noqa: E402
from engine.kernels.fused_attention import decode_attention_reference, fused_decode_attention  # noqa: E402

MODE = "interpreter" if os.environ.get("TRITON_INTERPRET") == "1" else "compiled-cuda"
DEV = torch.device("cuda" if MODE == "compiled-cuda" else "cpu")


def gemm_row(m, k, n, group=128, bias=False):
    g = torch.Generator().manual_seed(m * 7 + k + n)
    iw = torch.randint(0, 16, (k, n), generator=g)
    iz = torch.randint(0, 16, (k // group, n), generator=g)
    qw, qz = pack_awq(iw, iz, group)
    sc = (torch.rand(k // group, n, generator=g) * 0.02 + 0.002).half()
    x = torch.randn(m, k, generator=g).half()
    b = torch.randn(n, generator=g).half() if bias else None
    ref = awq_gemm_reference(x, qw, qz, sc, b, group)
    out = awq_gemm(x.to(DEV), qw.to(DEV), qz.to(DEV), sc.to(DEV), None if b is None else b.to(DEV),
                   group_size=group, out_dtype=torch.float32).cpu()
    return (out - ref).abs().max().item()


def attn_row(hq, hkv, d, s, cache_len):
    g = torch.Generator().manual_seed(s)
    q = (torch.randn(hq, d, generator=g) * 2).half()
    kc = torch.randn(hkv, cache_len, d, generator=g).half()
    vc = torch.randn(hkv, cache_len, d, generator=g).half()
    out = fused_decode_attention(q.to(DEV), kc.to(DEV), vc.to(DEV), s).cpu().float()
    return (out - decode_attention_reference(q, kc, vc, s)).abs().max().item()


def rms_row(rows, hidden, dtype):
    g = torch.Generator().manual_seed(rows + hidden)
    x = torch.randn(rows, hidden, generator=g).to(dtype)
    w = torch.randn(hidden, generator=g).to(dtype)
    return (rmsnorm(x, w).float() - rmsnorm_reference(x, w).float()).abs().max().item()


def main() -> int:
    rows = []
    for (m, k, n) in [(1, 2048, 2048), (1, 2048, 256), (1, 2048, 11008), (1, 11008, 2048), (4, 2048, 2048), (17, 2048, 2048)]:
        rows.append(("awq_gemm (fp32 out)", f"M={m} K={k} N={n}", "atol 1e-3", gemm_row(m, k, n, bias=(n in (256, 2048)))))
    for (hq, hkv, d, s, c) in [(16, 2, 128, 1, 64), (16, 2, 128, 64, 128), (16, 2, 128, 200, 256), (16, 2, 128, 512, 1024)]:
        rows.append(("fused_decode_attention", f"Hq={hq} Hkv={hkv} D={d} S={s}", "atol 1e-3", attn_row(hq, hkv, d, s, c)))
    for dtype in (torch.float32, torch.float16):
        for (r, h) in [(1, 2048), (8, 2048), (128, 2048)]:
            tol = "atol 1e-4" if dtype == torch.float32 else "fp16 rounding (1e-3 + 2e-3*|ref|)"
            rows.append((f"cpp rmsnorm ({str(dtype)[6:]})", f"rows={r} hidden={h}", tol, rms_row(r, h, dtype)))
    out = ROOT / "results" / "kernel_max_diff.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "shape", "mode", "tolerance", "max_abs_diff"])
        for op, shape, tol, d in rows:
            mode = "cpu-extension" if op.startswith("cpp") else MODE
            w.writerow([op, shape, mode, tol, f"{d:.3e}"])
            print(f"{op:26s} {shape:28s} {mode:14s} max|diff| {d:.3e}  ({tol})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
