#!/usr/bin/env python3
"""Roofline analysis of batch-1 decode for the engine's model.

Always (no GPU needed): derives FLOPs, bytes and arithmetic intensity (AI) of every
decode operation from the model configuration and writes
``results/roofline_analytic.json``.

With a CUDA GPU: also measures the machine's achievable DRAM bandwidth and fp16
matmul peak, and the achieved GB/s / GFLOP/s of the Triton kernels at the real
decode shapes (``results/hw_ceilings.json``, ``results/roofline_measured.json``).

Always: draws ``results/roofline.png``. The title says ANALYTIC when no
measurement exists. The analytic plot uses only (a) the datasheet-derived memory
bandwidth and (b) an illustrative FP32 non-tensor compute band built from
third-party clock figures; neither is a measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engine.model import QWEN2_5_3B, ModelConfig  # noqa: E402

DATASHEET_BW_GBS = 128 / 8 * 16  # 128-bit bus x 16 Gbps effective = 256 GB/s (third-party listing)
CUDA_CORES = 3072
CLOCK_GHZ_RANGE = (1.55, 2.37)  # third-party listing; depends on the laptop's TGP

# Validated categorical slots (dataviz skill), assigned to categories in fixed order.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
CAT_COLORS = {"4-bit GEMM": "#2a78d6", "lm_head fp16": "#eb6834", "decode attention": "#1baf7a",
              "RMSNorm": "#eda100"}


# --------------------------------------------------------------------------- #
# Analytic model                                                              #
# --------------------------------------------------------------------------- #
def linear_weight_bytes(k: int, n: int, group: int, bias: bool) -> int:
    """Bytes of one AWQ linear's tensors: int4 weights + fp16 scales + int4 zeros + fp16 bias."""
    return k * n // 2 + (k // group) * n * 2 + (k // group) * (n // 8) * 4 + (2 * n if bias else 0)


def linear_cost(k: int, n: int, group: int, bias: bool) -> tuple[float, float]:
    """(FLOPs, bytes) of one M=1 AWQ linear: its tensors plus the fp16 input and output vectors."""
    return 2.0 * k * n, float(linear_weight_bytes(k, n, group, bias) + 2 * k + 2 * n)


def analytic_model(cfg: ModelConfig, seq_len: int) -> dict:
    h, i, g = cfg.hidden_size, cfg.intermediate_size, cfg.group_size
    kv = cfg.num_kv_heads * cfg.head_dim
    layer_linears = {"q_proj": (h, h, True), "k_proj": (h, kv, True), "v_proj": (h, kv, True),
                     "o_proj": (h, h, False), "gate_proj": (h, i, False), "up_proj": (h, i, False),
                     "down_proj": (i, h, False)}
    ops = []
    for name, (k, n, bias) in layer_linears.items():
        f, b = linear_cost(k, n, g, bias)
        ops.append({"name": name, "category": "4-bit GEMM", "count": cfg.num_layers, "flops": f, "bytes": b})
    f = 2.0 * cfg.vocab_size * h
    b = cfg.vocab_size * h * 2 + 2 * h + 2 * cfg.vocab_size
    ops.append({"name": "lm_head", "category": "lm_head fp16", "count": 1, "flops": f, "bytes": b})
    f = 4.0 * cfg.num_heads * seq_len * cfg.head_dim
    b = 2 * cfg.num_kv_heads * seq_len * cfg.head_dim * 2 + 2 * cfg.num_heads * cfg.head_dim * 2
    ops.append({"name": f"attention (S={seq_len})", "category": "decode attention", "count": cfg.num_layers,
                "flops": f, "bytes": b})
    ops.append({"name": "rmsnorm", "category": "RMSNorm", "count": 2 * cfg.num_layers + 1,
                "flops": 4.0 * h, "bytes": 2 * h * 2 + 2 * h})
    for op in ops:
        op["ai"] = op["flops"] / op["bytes"]
    total_f = sum(o["flops"] * o["count"] for o in ops)
    total_b = sum(o["bytes"] * o["count"] for o in ops)
    kv_bytes = cfg.num_layers * 2 * cfg.num_kv_heads * cfg.head_dim * 2 * seq_len
    return {
        "model": "Qwen2.5-3B-Instruct-AWQ", "seq_len": seq_len, "ops": ops,
        "per_token": {
            "flops": total_f, "bytes": total_b, "ai": total_f / total_b,
            "kv_cache_bytes_read": kv_bytes,
            "min_time_ms_at_datasheet_bw": total_b / (DATASHEET_BW_GBS * 1e9) * 1e3,
            "max_tokens_per_s_at_datasheet_bw": DATASHEET_BW_GBS * 1e9 / total_b,
        },
        "ceilings": {
            "datasheet_bandwidth_GBs": DATASHEET_BW_GBS,
            "fp32_nontensor_peak_GFLOPs_range": [CUDA_CORES * 2 * c for c in CLOCK_GHZ_RANGE],
            "note": "datasheet bandwidth derived from a third-party listing (128-bit x 16 Gbps); "
                    "FP32 band = 3072 cores x 2 FLOP x third-party clock range. Neither is measured.",
        },
    }


# --------------------------------------------------------------------------- #
# Measurements (CUDA only)                                                    #
# --------------------------------------------------------------------------- #
def _time_cuda(fn, iters: int = 50, warmup: int = 10) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1e3  # seconds


def measure_ceilings() -> dict:
    import torch

    return {
        "device": torch.cuda.get_device_name(),
        "bandwidth_GBs_copy": _copy_bandwidth() / 1e9,
        "peak_fp16_matmul_GFLOPs": _matmul_flops() / 1e9,
        "datasheet_bandwidth_GBs": DATASHEET_BW_GBS,
    }


def _copy_bandwidth() -> float:
    import torch

    src = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    t = _time_cuda(lambda: dst.copy_(src), iters=20, warmup=3)
    return 2 * src.numel() / t  # a copy reads and writes every byte


def _matmul_flops() -> float:
    import torch

    a = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    b = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    t = _time_cuda(lambda: a @ b, iters=20, warmup=5)
    return 2 * 4096**3 / t


def measure_kernels(cfg: ModelConfig, seq_len: int) -> list[dict]:
    import torch

    sys.path.insert(0, str(ROOT))
    from engine.kernels.dequant_gemm import awq_gemm, pack_awq
    from engine.kernels.fused_attention import fused_decode_attention

    model = analytic_model(cfg, seq_len)
    by_name = {o["name"]: o for o in model["ops"]}
    out = []
    g = cfg.group_size
    shapes = {"q_proj": (cfg.hidden_size, cfg.hidden_size), "gate_proj": (cfg.hidden_size, cfg.intermediate_size),
              "down_proj": (cfg.intermediate_size, cfg.hidden_size)}
    for name, (k, n) in shapes.items():
        iw = torch.randint(0, 16, (k, n))
        iz = torch.randint(0, 16, (k // g, n))
        qw, qz = pack_awq(iw, iz, g)
        sc = (torch.rand(k // g, n) * 0.02 + 0.002).half()
        x = torch.randn(1, k, dtype=torch.float16, device="cuda")
        qw, qz, sc = qw.cuda(), qz.cuda(), sc.cuda()
        t = _time_cuda(lambda: awq_gemm(x, qw, qz, sc, group_size=g))
        op = by_name[name]
        out.append({"name": name, "category": "4-bit GEMM", "seconds": t, "ai": op["ai"],
                    "achieved_GFLOPs": op["flops"] / t / 1e9, "achieved_GBs": op["bytes"] / t / 1e9})
    q = torch.randn(cfg.num_heads, cfg.head_dim, dtype=torch.float16, device="cuda")
    kc = torch.randn(cfg.num_kv_heads, 2048, cfg.head_dim, dtype=torch.float16, device="cuda")
    vc = torch.randn_like(kc)
    t = _time_cuda(lambda: fused_decode_attention(q, kc, vc, seq_len))
    op = by_name[f"attention (S={seq_len})"]
    out.append({"name": op["name"], "category": "decode attention", "seconds": t, "ai": op["ai"],
                "achieved_GFLOPs": op["flops"] / t / 1e9, "achieved_GBs": op["bytes"] / t / 1e9})
    return out


# --------------------------------------------------------------------------- #
# Plot                                                                        #
# --------------------------------------------------------------------------- #
def plot(analytic: dict, ceilings: dict | None, measured: list[dict] | None, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    bw = (ceilings["bandwidth_GBs_copy"] if ceilings else DATASHEET_BW_GBS)  # GB/s == GFLOP/s per FLOP/B
    xs = np.logspace(-1, 3, 200)
    ax.plot(xs, bw * xs, color=INK, lw=2, zorder=3)
    ax.text(0.13, bw * 0.13 * 1.35, f"memory roof: {bw:.0f} GB/s " + ("(measured copy)" if ceilings else "(datasheet-derived)"),
            rotation=33, rotation_mode="anchor", fontsize=9, color=INK2)

    if ceilings:
        peak = ceilings["peak_fp16_matmul_GFLOPs"]
        ax.axhline(peak, color=INK, lw=2, zorder=3)
        ax.text(250, peak * 1.12, f"fp16 matmul peak (measured): {peak / 1e3:.1f} TFLOP/s", ha="right", fontsize=9, color=INK2)
        ridge = peak / bw
    else:
        lo, hi = analytic["ceilings"]["fp32_nontensor_peak_GFLOPs_range"]
        ax.axhspan(lo, hi, color=GRID, zorder=1)
        ax.text(0.12, hi * 1.08, "FP32 non-tensor peak range (third-party clocks, not measured)", ha="left",
                fontsize=9, color=INK2)
        ridge = (lo / bw, hi / bw)
        ax.axvspan(ridge[0], ridge[1], color=GRID, alpha=0.6, zorder=0)
        ax.text(np.sqrt(ridge[0] * ridge[1]), 14, f"ridge point\n{ridge[0]:.0f}-{ridge[1]:.0f} FLOP/B",
                ha="center", fontsize=9, color=INK2)

    seen = set()
    for op in analytic["ops"]:
        cat = op["category"]
        ai = op["ai"]
        attainable = bw * ai
        label = cat if cat not in seen else None
        seen.add(cat)
        ax.scatter([ai], [attainable], s=70, color=CAT_COLORS[cat], edgecolor=SURFACE, linewidth=1.5,
                   zorder=5, label=label if not measured else None)
    offsets = {"4-bit GEMM": (10, -16), "lm_head fp16": (10, -16), "decode attention": (10, -16),
               "RMSNorm": (10, -16)}
    for cat, off in offsets.items():
        op = next(o for o in analytic["ops"] if o["category"] == cat)
        ax.annotate(f"{cat}  (AI {op['ai']:.2f})", xy=(op["ai"], bw * op["ai"]), xytext=off,
                    textcoords="offset points", fontsize=9, color=INK, va="top")

    if measured:
        for m in measured:
            ax.scatter([m["ai"]], [m["achieved_GFLOPs"]], s=70, marker="D", color=CAT_COLORS[m["category"]],
                       edgecolor=INK, linewidth=1.0, zorder=6)
            ax.text(m["ai"] * 1.1, m["achieved_GFLOPs"] * 0.78, f"{m['name']}: {m['achieved_GBs']:.0f} GB/s",
                    fontsize=8, color=INK2)
        ax.scatter([], [], marker="D", color=INK2, label="measured kernel (achieved)")
        ax.scatter([], [], marker="o", color=INK2, label="attainable at the memory roof")
        ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK2)
    else:
        ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK2, title="attainable at the memory roof")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(8, 4e4)
    ax.set_xlabel("arithmetic intensity (FLOP per byte of DRAM traffic)")
    ax.set_ylabel("performance (GFLOP/s)")
    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    mode = f"MEASURED on {ceilings['device']}" if ceilings else "ANALYTIC (first principles; no GPU measurement)"
    ax.set_title(f"Roofline of batch-1 decode, Qwen2.5-3B AWQ int4 - {mode}", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--results", default=str(ROOT / "results"))
    args = ap.parse_args()
    res = Path(args.results)
    res.mkdir(parents=True, exist_ok=True)

    analytic = analytic_model(QWEN2_5_3B, args.seq_len)
    (res / "roofline_analytic.json").write_text(json.dumps(analytic, indent=2))
    ceilings = measured = None
    try:
        import torch

        if torch.cuda.is_available():
            ceilings = measure_ceilings()
            measured = measure_kernels(QWEN2_5_3B, args.seq_len)
            (res / "hw_ceilings.json").write_text(json.dumps(ceilings, indent=2))
            (res / "roofline_measured.json").write_text(json.dumps(measured, indent=2))
    except ImportError:
        pass
    if ceilings is None:
        print("no CUDA device: wrote the ANALYTIC roofline only (no measured ceilings or kernel points)")
    plot(analytic, ceilings, measured, res / "roofline.png")
    pt = analytic["per_token"]
    print(f"per token: {pt['flops'] / 1e9:.2f} GFLOP, {pt['bytes'] / 1e9:.3f} GB, AI {pt['ai']:.2f} FLOP/B; "
          f"memory-bound ceiling at {DATASHEET_BW_GBS:.0f} GB/s: {pt['max_tokens_per_s_at_datasheet_bw']:.1f} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
