#!/usr/bin/env python3
"""Speedup of the C++ RMSNorm over the PyTorch reference, per device.

Writes results/cpp_op_bench.csv (one row per device/dtype/shape). The device column
says where each number was measured; CPU rows and CUDA rows are different claims:
  * CPU:  hand-written single-pass C++ kernel vs PyTorch eager (5+ separate ops).
  * CUDA: C++ composition of the same ATen kernels vs the Python-level reference,
          i.e. only Python dispatch overhead is removed (see src/csrc/custom_ops.cpp).
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from engine.cpp_ops import load_custom_ops, rmsnorm, rmsnorm_reference  # noqa: E402

SHAPES = [(1, 2048), (8, 2048), (128, 2048)]
DTYPES = [torch.float32, torch.float16]


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown CPU"


def time_fn(fn, device: torch.device, repeats: int = 9, target_s: float = 0.05) -> float:
    """Median seconds per call."""
    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)
    for _ in range(20):
        fn()
    sync()
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < 0.01:
        fn()
        n += 1
    sync()
    iters = max(10, int(target_s / ((time.perf_counter() - t0) / n)))
    samples = []
    for _ in range(repeats):
        sync()
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        sync()
        samples.append((time.perf_counter() - t) / iters)
    return statistics.median(samples)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "cpp_op_bench.csv"))
    ap.add_argument("--repeats", type=int, default=4,
                    help="independent measurement passes; timings of PyTorch's own multi-row fp16 kernel "
                         "vary between passes, so a single pass is not reported")
    args = ap.parse_args()
    load_custom_ops()
    devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    rows = []
    for rep in range(1, args.repeats + 1):
        for dev in devices:
            label = f"cpu: {cpu_name()}" if dev.type == "cpu" else torch.cuda.get_device_name()
            for dtype in DTYPES:
                for shape in SHAPES:
                    x = torch.randn(*shape, device=dev).to(dtype)
                    w = torch.randn(shape[-1], device=dev).to(dtype)
                    ref = time_fn(lambda: rmsnorm_reference(x, w), dev)
                    cpp = time_fn(lambda: rmsnorm(x, w), dev)
                    rows.append({"run": rep, "device": label,
                                 "torch_threads": torch.get_num_threads() if dev.type == "cpu" else "",
                                 "dtype": str(dtype).replace("torch.", ""), "shape": "x".join(map(str, shape)),
                                 "pytorch_reference_us": ref * 1e6, "cpp_op_us": cpp * 1e6, "speedup": ref / cpp})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    for r in rows:
        print(f"run {r['run']} {r['device'][:34]:34s} {r['dtype']:8s} {r['shape']:9s} ref {r['pytorch_reference_us']:8.1f} us  "
              f"cpp {r['cpp_op_us']:8.1f} us  speedup {r['speedup']:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
