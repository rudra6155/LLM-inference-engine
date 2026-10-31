#!/usr/bin/env python3
"""Bar charts of results/benchmark.csv -> results/throughput.png.

One small panel per metric (each with its own honest zero-based axis; no dual
axes). Bars use a fixed color per backend, never reassigned by rank. Every bar is
labeled with its value, which also satisfies the contrast relief rule for the
lighter palette slots. Only real rows of the CSV are plotted.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
# Validated categorical slots (dataviz skill), fixed per entity.
COLORS = {"engine": "#2a78d6", "llama.cpp": "#eb6834", "vllm": "#1baf7a", "engine-reference-ops": "#eda100",
          "engine-graph": "#6250d6", "engine-tuned": "#4e9aa8", "engine-graph-tuned": "#c13a93"}
ORDER = ["engine", "engine-graph", "engine-tuned", "engine-graph-tuned", "llama.cpp", "vllm", "engine-reference-ops"]
LABELS = {"engine": "this engine", "llama.cpp": "llama.cpp", "vllm": "vLLM",
          "engine-reference-ops": "engine, PyTorch ops", "engine-graph": "engine\n+ CUDA graph",
          "engine-tuned": "engine\n+ tuned", "engine-graph-tuned": "engine + graph\n+ tuned"}
PANELS = [
    ("decode_tok_s", "Decode tokens/s", "higher is better", "{:.1f}"),
    ("itl_ms_p99", "p99 inter-token latency (ms)", "lower is better", "{:.1f}"),
    ("ttft_ms_p50", "TTFT, median (ms)", "lower is better", "{:.0f}"),
    ("peak_vram_mib", "Peak VRAM above baseline (MiB)", "lower is better", "{:.0f}"),
]


def load_latest(csv_path: Path) -> dict[str, dict]:
    rows = list(csv.DictReader(csv_path.open()))
    latest: dict[str, dict] = {}
    for r in rows:  # later rows override earlier ones for the same backend
        latest[r["backend"]] = r
    return latest


def plot(latest: dict[str, dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    backends = [b for b in ORDER if b in latest] + [b for b in latest if b not in ORDER]
    ref = latest[backends[0]]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), dpi=160, facecolor=SURFACE)
    for ax, (col, title, note, fmt) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        vals = []
        for b in backends:
            raw = latest[b].get(col, "")
            vals.append(float(raw) if raw not in ("", None) else None)
        xs = range(len(backends))
        top = max([v for v in vals if v is not None] or [1.0])
        for x, b, v in zip(xs, backends, vals):
            if v is None:
                ax.text(x, top * 0.02, "n/a", ha="center", va="bottom", color=INK2, fontsize=9)
                continue
            ax.bar(x, v, width=0.62, color=COLORS.get(b, "#6250d6"), zorder=3)
            ax.text(x, v + top * 0.015, fmt.format(v), ha="center", va="bottom", color=INK, fontsize=9)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([LABELS.get(b, b) for b in backends], fontsize=8.5)
        ax.set_ylim(0, top * 1.14)
        ax.set_title(f"{title}  -  {note}", fontsize=10, loc="left", color=INK)
        ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle(
        f"Batch-size-1 decode, Qwen2.5-3B 4-bit on {ref['device']}\n"
        f"{ref['prompt_tokens']} prompt tokens, {ref['new_tokens']} new, {ref['runs']} runs, greedy",
        fontsize=11, x=0.02, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "results" / "benchmark.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "throughput.png"))
    args = ap.parse_args(argv)
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"{csv_path} does not exist; run `make benchmark` first (nothing plotted)", file=sys.stderr)
        return 1
    latest = load_latest(csv_path)
    if not latest:
        print(f"{csv_path} has no rows (nothing plotted)", file=sys.stderr)
        return 1
    plot(latest, Path(args.out))
    print(f"wrote {args.out} from {len(latest)} backend row(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
