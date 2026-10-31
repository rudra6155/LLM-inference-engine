#!/usr/bin/env python3
"""Profile batch-1 decode steps of one engine variant.

``--variant custom``     Triton fused dequant+GEMM, Triton decode attention, C++ RMSNorm.
``--variant reference``  The same weights run through PyTorch ops (dequantize one weight
                         matrix at a time, then matmul; SDPA; PyTorch RMSNorm). This is
                         the profiling baseline for "what the custom kernels replace".

Default mode: ``torch.profiler`` over ``--steps`` decode steps. Writes
``results/traces/decode_<variant>.json`` (Chrome trace; open in Perfetto) and
``results/profile_<variant>_kernels.csv`` / ``..._summary.json``.

``--ncu`` mode: no torch profiler; the decode steps are bracketed with
``cudaProfilerStart/Stop`` so that Nsight Compute launched with
``--profile-from-start off`` measures only those kernels (see ``profile.sh``).

Requires a CUDA GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CATEGORIES = [  # (category, substrings of kernel names), first match wins
    ("custom: fused dequant+GEMM", ("_awq_gemm_kernel", "_splitk_reduce_kernel")),
    ("custom: fused decode attention", ("_decode_attn_kernel",)),
    ("GEMM / GEMV (cuBLAS, incl. lm_head)", ("gemm", "gemv", "cutlass", "sgemm", "cublas", "ampere", "sm80", "sm89")),
    ("attention (PyTorch SDPA)", ("flash", "fmha", "sdpa", "attention", "softmax")),
    ("dequantize / unpack (PyTorch elementwise)", ("bitwise", "rshift", "shift", "to_copy", "copy", "mul", "sub", "repeat", "index")),
    ("normalization / activation", ("rms", "pow", "mean", "rsqrt", "silu", "reduce")),
]


def categorize(name: str) -> str:
    low = name.lower()
    for cat, keys in CATEGORIES:
        if any(k.lower() in low for k in keys):
            return cat
    return "other"


def kernels_from_chrome_trace(trace_path: Path) -> list[dict]:
    """CUDA kernel events (category ``kernel``) from a Kineto Chrome trace."""
    events = json.loads(trace_path.read_text())["traceEvents"]
    return [{"name": e["name"], "dur_us": float(e["dur"])} for e in events
            if e.get("cat") == "kernel" and "dur" in e]


def summarize_kernels(kernels: list[dict], steps: int) -> list[dict]:
    agg: dict[str, dict] = {}
    for k in kernels:
        a = agg.setdefault(k["name"], {"name": k["name"], "category": categorize(k["name"]), "calls": 0, "total_us": 0.0})
        a["calls"] += 1
        a["total_us"] += k["dur_us"]
    rows = sorted(agg.values(), key=lambda r: -r["total_us"])
    for r in rows:
        r["avg_us"] = r["total_us"] / r["calls"]
        r["calls_per_step"] = r["calls"] / steps
    return rows


def build_summary(*, variant: str, steps: int, device: str, rows: list[dict], wall_s: float,
                  useful_bytes: float, copy_bw_gbs: float | None, timing: dict | None = None) -> dict:
    """Assemble the per-variant summary. Never divides by zero.

    ``rows`` come from the torch.profiler Chrome trace (CUPTI). If CUPTI recorded no kernels
    (``busy_ms == 0``) every figure derived from kernel time is ``None`` and a warning says so;
    the CUPTI-free ``timing`` figures (eager / CUDA-graph replay) are still reported.
    """
    timing = timing or {}
    warnings: list[str] = []
    busy_ms = sum(r["total_us"] for r in rows) / steps / 1e3
    launches = sum(r["calls"] for r in rows) / steps
    if busy_ms <= 0:
        warnings.append(
            "torch.profiler recorded no CUDA kernel events, so GPU-busy time, the kernel launch count and "
            "the bandwidth derived from them are unavailable. CUPTI failing to initialise is a known "
            "problem on WSL2 (check the output for 'CUPTI initialization failed'), but that is not proven "
            "here. eager_ms_per_step / graph_replay_ms_per_step below do not use CUPTI.")
        busy_ms_out = launches_out = None
    else:
        busy_ms_out, launches_out = busy_ms, launches

    def bandwidth(ms: float | None) -> float | None:
        return None if not ms or ms <= 0 else useful_bytes / (ms / 1e3) / 1e9

    def pct(gbs: float | None) -> float | None:
        return None if gbs is None or not copy_bw_gbs else 100 * gbs / copy_bw_gbs

    bw_busy = bandwidth(busy_ms_out)
    graph_ms = timing.get("graph_replay_ms_per_step")
    bw_graph = bandwidth(graph_ms)
    eager_ms = timing.get("eager_ms_per_step")
    return {
        "variant": variant, "steps": steps, "device": device,
        "gpu_busy_ms_per_step": busy_ms_out,
        "wall_ms_per_step_under_profiler": wall_s * 1e3,
        "kernel_launches_per_step": launches_out,
        "useful_bytes_per_step": useful_bytes,
        "useful_bandwidth_GBs": bw_busy,
        "measured_copy_bandwidth_GBs": copy_bw_gbs,
        "useful_bandwidth_pct_of_measured_copy_bw": pct(bw_busy),
        "eager_ms_per_step": eager_ms,
        "graph_replay_ms_per_step": graph_ms,
        "launch_overhead_fraction_of_eager": (None if not eager_ms or not graph_ms else max(0.0, 1 - graph_ms / eager_ms)),
        "useful_bandwidth_GBs_graph": bw_graph,
        "useful_bandwidth_pct_of_measured_copy_bw_graph": pct(bw_graph),
        "timing_errors": {k: v for k, v in timing.items() if k.endswith("_error")},
        "warnings": warnings,
        "note": "useful bytes = weights + KV that must be read per token (analytic); the same numerator for "
                "both variants. All bandwidths are derived (bytes / time), not hardware counters. "
                "graph_replay_ms_per_step = GPU execution time of ONE captured decode step replayed as a CUDA "
                "graph, i.e. with the per-kernel launch / Python overhead removed; it is a measurement "
                "instrument here, not a feature of the engine.",
    }


def measure_launch_vs_gpu(step, graph_step, steps: int) -> dict:
    """CUPTI-free timing of one decode step.

    ``eager_ms_per_step``: ``steps`` back-to-back eager steps (no per-token host sync), as the engine runs.
    ``graph_replay_ms_per_step``: one step captured into a CUDA graph and replayed - GPU time with the
    launch overhead removed. If capture fails (e.g. an op that syncs or copies host->device), the reason
    is returned in ``graph_capture_error`` instead of raising.
    """
    import torch

    out: dict = {}
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
        out["eager_ms_per_step"] = (time.perf_counter() - t0) / steps * 1e3
    except Exception as e:  # noqa: BLE001 - diagnostics must never kill the run
        out["eager_error"] = f"{type(e).__name__}: {e}"
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm-up on a side stream before capture, as PyTorch requires
            for _ in range(3):
                graph_step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_step()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        replays = max(steps, 10)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        out["graph_replay_ms_per_step"] = start.elapsed_time(end) / replays
    except Exception as e:  # noqa: BLE001
        out["graph_capture_error"] = f"{type(e).__name__}: {e}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["custom", "reference"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct-AWQ")
    ap.add_argument("--loader", default="autoawq", choices=["autoawq", "safetensors"])
    ap.add_argument("--prompt-file", default=str(ROOT / "results" / "prompt.json"))
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--ncu", action="store_true")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--tuned-config", help="file from scripts/tune_kernels.py (custom variant only)")
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("profile_decode.py requires a CUDA GPU", file=sys.stderr)
        return 2
    from engine.generate import KVCache
    from engine.model import EngineOptions, load_awq_model
    from roofline import analytic_model

    prompt_path = Path(args.prompt_file)
    ids = json.loads(prompt_path.read_text())["ids"] if prompt_path.exists() else list(range(1000, 1128))
    opts = (EngineOptions("reference", "reference", "reference") if args.variant == "reference"
            else EngineOptions(tuned_config=args.tuned_config))
    model = load_awq_model(args.model, loader=args.loader, device="cuda", max_seq_len=args.max_seq_len, options=opts)
    cache = KVCache(model.cfg, model.max_seq_len, model.device)

    prompt = torch.tensor(ids, device="cuda")
    tok = torch.argmax(model.prefill(prompt, cache))
    pos = len(ids)

    def step():
        nonlocal tok, pos
        tok = torch.argmax(model.decode_step(tok, cache, pos))
        pos += 1

    for _ in range(args.warmup):  # triggers Triton compilation / autotune and cuBLAS init
        step()
    torch.cuda.synchronize()

    if args.ncu:
        torch.cuda.profiler.start()
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        return 0

    from torch.profiler import ProfilerActivity, profile

    res = Path(args.results)
    (res / "traces").mkdir(parents=True, exist_ok=True)
    trace_path = res / "traces" / f"decode_{args.variant}.json"
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / args.steps  # includes profiler overhead; indicative only
    prof.export_chrome_trace(str(trace_path))

    rows = summarize_kernels(kernels_from_chrome_trace(trace_path), args.steps)
    with (res / f"profile_{args.variant}_kernels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "category", "calls", "calls_per_step", "total_us", "avg_us"])
        w.writeheader()
        w.writerows({k: r[k] for k in w.fieldnames} for r in rows)

    analytic = analytic_model(model.cfg, pos)["per_token"]
    ceilings_path = res / "hw_ceilings.json"
    peak = json.loads(ceilings_path.read_text())["bandwidth_GBs_copy"] if ceilings_path.exists() else None

    tok_static, pos_fixed = tok.clone(), pos  # the captured step reads a fixed token at a fixed position

    def graph_step():
        torch.argmax(model.decode_step(tok_static, cache, pos_fixed))

    timing = measure_launch_vs_gpu(step, graph_step, args.steps)
    summary = build_summary(variant=args.variant, steps=args.steps, device=torch.cuda.get_device_name(),
                            rows=rows, wall_s=wall, useful_bytes=analytic["bytes"], copy_bw_gbs=peak,
                            timing=timing)
    for w in summary["warnings"]:
        print(f"WARNING: {w}", file=sys.stderr)
    (res / f"profile_{args.variant}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
