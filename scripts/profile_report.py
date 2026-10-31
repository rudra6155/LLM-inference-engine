#!/usr/bin/env python3
"""Combine profiler outputs into results/profile_summary.csv and results/dram_throughput.png.

Inputs (produced by profile.sh on a GPU machine):
  results/profile_<variant>_summary.json   torch.profiler derived numbers
  results/profile_<variant>_kernels.csv    per-kernel GPU time from the Chrome trace
  results/logs/ncu_<variant>.csv           Nsight Compute raw CSV (optional; needs `ncu`)

``--print-ncu-metrics`` prints the metric list to pass to ``ncu --metrics``: four core
metrics always, plus stall metrics only when ``ncu --query-metrics`` lists their base
name on this machine.

Nothing here invents data: with no ncu output the DRAM-counter panel is omitted and
the report says so.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from profile_decode import categorize  # noqa: E402

CORE_METRICS = [
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
]
# Offered only if `ncu --query-metrics` lists the base name on this machine.
STALL_CANDIDATES = [
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
]
TIME, READ, WRITE, PCT = CORE_METRICS


def ncu_metric_list() -> list[str]:
    metrics = list(CORE_METRICS)
    if shutil.which("ncu"):
        out = subprocess.run(["ncu", "--query-metrics"], capture_output=True, text=True).stdout
        known = {line.split()[0] for line in out.splitlines() if line.strip()}
        metrics += [m for m in STALL_CANDIDATES if m.split(".")[0] in known]
    return metrics


def _num(s: str) -> float:
    return float(s.replace(",", "").strip())


def parse_ncu_raw_csv(text: str) -> tuple[list[dict], dict[str, str]]:
    """Rows (one per profiled kernel launch) and the units row, from `ncu --csv --page raw`.

    Expected layout: optional ``==PROF==`` log lines, a header line starting with ``"ID"``,
    an optional units row, then one row per kernel with a numeric ID.
    """
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    if start is None:
        head = "\n".join(lines[:15])
        raise ValueError(f"no CSV header starting with \"ID\" found in ncu output. First lines:\n{head}")
    reader = csv.reader(lines[start:])
    header = next(reader)
    rows, units = [], {}
    for rec in reader:
        if not rec:
            continue
        d = dict(zip(header, rec))
        if rec[0].strip().isdigit():
            rows.append(d)
        elif not units:
            units = d
    if "Kernel Name" not in header:
        raise ValueError(f"ncu CSV has no 'Kernel Name' column; columns: {header}")
    return rows, units


def aggregate_ncu(rows: list[dict], units: dict[str, str], steps: int) -> dict[str, dict]:
    """Per-category sums (per decode step) and time-weighted mean DRAM throughput %."""
    for col, want in ((TIME, "nsecond"), (READ, "byte"), (WRITE, "byte")):
        if col in units and units[col] and units[col] != want:
            raise ValueError(f"unexpected unit {units[col]!r} for {col}; run ncu with --print-units base")
    cats: dict[str, dict] = {}
    for r in rows:
        c = cats.setdefault(categorize(r["Kernel Name"]), {"launches": 0, "time_ns": 0.0, "read": 0.0,
                                                           "write": 0.0, "pct_x_time": 0.0, "stall_x_time": 0.0})
        t = _num(r[TIME])
        c["launches"] += 1
        c["time_ns"] += t
        c["read"] += _num(r[READ])
        c["write"] += _num(r[WRITE])
        c["pct_x_time"] += _num(r[PCT]) * t
        stall = next((_num(r[m]) for m in STALL_CANDIDATES if m in r and r[m]), None)
        if stall is not None:
            c["stall_x_time"] += stall * t
    out = {}
    for cat, c in cats.items():
        out[cat] = {
            "launches_per_step": c["launches"] / steps,
            "gpu_time_us_per_step": c["time_ns"] / 1e3 / steps,
            "dram_read_MB_per_step": c["read"] / 1e6 / steps,
            "dram_write_MB_per_step": c["write"] / 1e6 / steps,
            "dram_throughput_pct_time_weighted": c["pct_x_time"] / c["time_ns"] if c["time_ns"] else 0.0,
            "stall_metric_time_weighted": c["stall_x_time"] / c["time_ns"] if c["time_ns"] and c["stall_x_time"] else None,
        }
    return out


def totals(agg: dict[str, dict]) -> dict:
    t = sum(v["gpu_time_us_per_step"] for v in agg.values())
    return {
        "gpu_time_us_per_step": t,
        "dram_MB_per_step": sum(v["dram_read_MB_per_step"] + v["dram_write_MB_per_step"] for v in agg.values()),
        "dram_throughput_pct_time_weighted":
            sum(v["dram_throughput_pct_time_weighted"] * v["gpu_time_us_per_step"] for v in agg.values()) / t if t else 0.0,
    }


def _f(v, spec: str) -> str:
    """Format ``v`` with ``spec``; ``None`` (a measurement that does not exist) prints as n/a."""
    return "n/a" if v is None else format(v, spec)


def _pct(fraction):
    return None if fraction is None else 100 * fraction


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--ncu-steps", type=int, default=2, help="decode steps captured by ncu")
    ap.add_argument("--print-ncu-metrics", action="store_true")
    args = ap.parse_args(argv)
    if args.print_ncu_metrics:
        print(",".join(ncu_metric_list()))
        return 0

    res = Path(args.results)
    variants = [v for v in ("custom", "reference") if (res / f"profile_{v}_summary.json").exists()]
    if not variants:
        print("no profile_*_summary.json found; run profile.sh on a GPU machine first", file=sys.stderr)
        return 1
    ncu: dict[str, dict] = {}
    for v in variants:
        p = res / "logs" / f"ncu_{v}.csv"
        if p.exists() and p.stat().st_size:
            rows, units = parse_ncu_raw_csv(p.read_text())
            ncu[v] = aggregate_ncu(rows, units, args.ncu_steps)

    with (res / "profile_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "category", "launches_per_step", "gpu_time_us_per_step", "dram_read_MB_per_step",
                    "dram_write_MB_per_step", "dram_throughput_pct_time_weighted", "stall_metric_time_weighted"])
        for v, agg in ncu.items():
            for cat, a in sorted(agg.items(), key=lambda kv: -kv[1]["gpu_time_us_per_step"]):
                w.writerow([v, cat] + [("" if a[k] is None else f"{a[k]:.4f}") for k in
                           ("launches_per_step", "gpu_time_us_per_step", "dram_read_MB_per_step",
                            "dram_write_MB_per_step", "dram_throughput_pct_time_weighted", "stall_metric_time_weighted")])
    for v in variants:
        s = json.loads((res / f"profile_{v}_summary.json").read_text())
        line = (f"{v}: GPU busy {_f(s.get('gpu_busy_ms_per_step'), '.2f')} ms/step, "
                f"{_f(s.get('kernel_launches_per_step'), '.0f')} launches/step, "
                f"useful bandwidth {_f(s.get('useful_bandwidth_GBs'), '.1f')} GB/s")
        if s.get("eager_ms_per_step") is not None or s.get("graph_replay_ms_per_step") is not None:
            line += (f"\n    CUPTI-free: eager {_f(s.get('eager_ms_per_step'), '.2f')} ms/step, CUDA-graph replay "
                     f"{_f(s.get('graph_replay_ms_per_step'), '.2f')} ms/step (GPU time, launch overhead removed); "
                     f"launch overhead = {_f(_pct(s.get('launch_overhead_fraction_of_eager')), '.0f')}% of eager; "
                     f"useful bandwidth at graph speed {_f(s.get('useful_bandwidth_GBs_graph'), '.1f')} GB/s "
                     f"= {_f(s.get('useful_bandwidth_pct_of_measured_copy_bw_graph'), '.0f')}% of measured copy bandwidth")
        for e in s.get("timing_errors", {}).items():
            line += f"\n    {e[0]}: {e[1]}"
        for w in s.get("warnings", []):
            line += f"\n    WARNING: {w}"
        if v in ncu:
            t = totals(ncu[v])
            line += f"; ncu: DRAM {t['dram_throughput_pct_time_weighted']:.1f}% of peak (time-weighted), {t['dram_MB_per_step']:.0f} MB/step"
        print(line)
    if not ncu:
        print("note: no Nsight Compute output (results/logs/ncu_<variant>.csv); DRAM-throughput counters are "
              "unavailable, only derived useful bandwidth is reported", file=sys.stderr)
    if not ncu and not any(float(r["total_us"]) > 0 for v in variants
                           for r in csv.DictReader((res / f"profile_{v}_kernels.csv").open())):
        print("note: no kernel timing exists for any variant (CUPTI recorded nothing), so dram_throughput.png "
              "is not drawn; see the CUPTI-free eager / CUDA-graph figures above", file=sys.stderr)
        return 0
    plot(res, variants, ncu)
    return 0


def plot(res: Path, variants: list[str], ncu: dict[str, dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    colors = {"custom": "#2a78d6", "reference": "#eb6834"}
    names = {"custom": "custom kernels", "reference": "PyTorch ops (baseline)"}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    panels = [("gpu_time_us_per_step", "GPU time per decode step by kernel category (us)", "{:.0f}"),
              ("dram_throughput_pct_time_weighted", "DRAM throughput, % of peak (time-weighted, Nsight Compute)", "{:.0f}")]
    if not ncu:
        panels = panels[:1]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9.5, 4.2 * len(panels)), dpi=160, facecolor=SURFACE, squeeze=False)
    src = ncu if ncu else None
    if src is None:  # derive per-category time from the torch.profiler kernel CSVs
        src = {}
        for v in variants:
            agg: dict[str, dict] = {}
            steps = json.loads((res / f"profile_{v}_summary.json").read_text())["steps"]
            for r in csv.DictReader((res / f"profile_{v}_kernels.csv").open()):
                a = agg.setdefault(r["category"], {"gpu_time_us_per_step": 0.0})
                a["gpu_time_us_per_step"] += float(r["total_us"]) / steps
            src[v] = agg
    TOTAL = "ALL KERNELS (total)"
    cats = sorted({c for v in src for c in src[v]}, key=lambda c: -max(src[v].get(c, {}).get("gpu_time_us_per_step", 0) for v in src))
    for v in src:  # like-for-like per-variant total, since the variants run different kernels
        tot = {"gpu_time_us_per_step": sum(a["gpu_time_us_per_step"] for a in src[v].values())}
        if ncu:
            tot["dram_throughput_pct_time_weighted"] = totals(src[v])["dram_throughput_pct_time_weighted"]
        src[v] = {TOTAL: tot, **src[v]}
    cats = [TOTAL] + cats
    for ax, (key, title, fmt) in zip(axes.flat, panels):
        ax.set_facecolor(SURFACE)
        n = len(src)
        width = 0.8 / n
        for j, v in enumerate(src):
            vals = [src[v].get(c, {}).get(key, 0.0) or 0.0 for c in cats]
            xs = np.arange(len(cats)) + (j - (n - 1) / 2) * width
            ax.bar(xs, vals, width=width * 0.92, color=colors[v], label=names[v], zorder=3)
            for x, val in zip(xs, vals):
                if val:
                    ax.text(x, val, fmt.format(val), ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([c.replace(" (", "\n(") for c in cats], fontsize=8)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    if not ncu:
        fig.text(0.02, 0.01, "Nsight Compute output not available: DRAM-throughput counters not shown.", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(res / "dram_throughput.png", facecolor=SURFACE)
    print(f"wrote {res / 'dram_throughput.png'}")


if __name__ == "__main__":
    sys.exit(main())
