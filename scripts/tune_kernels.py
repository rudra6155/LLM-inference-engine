#!/usr/bin/env python3
"""Measure launch configurations for the decode kernels on the target GPU and write the winners.

What is swept
  * ``awq_gemm`` for every decode shape (M = 1) of Qwen2.5-3B: tile ``BLOCK_N`` / ``BLOCK_K``, ``SPLIT_K``,
    ``num_warps``, ``num_stages``. A wider ``BLOCK_N`` makes the packed-weight loads contiguous over more bytes
    per row (``BLOCK_N / 2`` bytes: 32 B at the default 64, 128 B at 256) - the memory-coalescing knob.
  * ``fused_decode_attention``: ``block_n``, ``num_warps``, ``num_stages`` and ``num_splits`` (split-KV; the
    unsplit kernel launches only ``H_kv`` = 2 programs on a GPU with many SMs).

How it is measured (this is what makes the numbers worth reading)
  * Every candidate is checked against a reference first (atol 1e-3 for correctness); a candidate that
    disagrees can never win.
  * Timing is of calls replayed from a **CUDA graph**, so Python / launch overhead (81% of an eager decode step
    here) cannot pollute the comparison between configurations.
  * Calls cycle through **many distinct copies** of the weights (or KV caches), a working set several times the
    L2 size, so the kernel reads DRAM instead of hitting cache - decode reads each weight once per token.
    (``results/roofline_measured.json`` reused one matrix per kernel; the 11.7 MB gate_proj fits in the 4060's
    L2, so its bandwidth figure may be flattered.)
  * The default configuration is measured the same way; a tuned config replaces it only if it is faster by more
    than ``--min-gain`` (default 3%), so noise does not rewrite the defaults.

Output: ``<out-dir>/best_configs.json`` (read back with ``EngineOptions(tuned_config=...)``), plus the full sweep in
``awq_gemm_sweep.csv`` / ``attention_sweep.csv``. Needs a CUDA GPU; ``--smoke`` runs a tiny CPU version through the
Triton interpreter only to exercise this script's logic (its timings are meaningless).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import itertools
import math
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GEMM_SPACES = {
    "quick": dict(BLOCK_N=(64, 128, 256), BLOCK_K=(64, 128), SPLIT_K=(1, 2, 4, 8, 16), num_warps=(4,), num_stages=(3,)),
    "standard": dict(BLOCK_N=(64, 128, 256), BLOCK_K=(64, 128), SPLIT_K=(1, 2, 4, 8, 16), num_warps=(4, 8),
                     num_stages=(2, 3, 4)),
    "full": dict(BLOCK_N=(64, 128, 256), BLOCK_K=(32, 64, 128), SPLIT_K=(1, 2, 4, 8, 16), num_warps=(2, 4, 8),
                 num_stages=(2, 3, 4)),
    "smoke": dict(BLOCK_N=(64, 128), BLOCK_K=(64,), SPLIT_K=(1, 2), num_warps=(4,), num_stages=(2,)),
}
ATTN_SPACES = {
    "quick": dict(block_n=(32, 64, 128), num_warps=(4,), num_stages=(3,), num_splits=(1, 2, 4, 8)),
    "standard": dict(block_n=(32, 64, 128), num_warps=(2, 4, 8), num_stages=(2, 3, 4), num_splits=(1, 2, 4, 8)),
    "full": dict(block_n=(32, 64, 128), num_warps=(2, 4, 8), num_stages=(2, 3, 4), num_splits=(1, 2, 4, 8, 16)),
    "smoke": dict(block_n=(32, 64), num_warps=(4,), num_stages=(2,), num_splits=(1, 2)),
}


# --------------------------------------------------------------------------- #
# Pure logic (unit-tested)                                                     #
# --------------------------------------------------------------------------- #
def gemm_candidates(k: int, n: int, group_size: int, space: dict, block_m: int = 16) -> list[dict]:
    """All launch configs of ``space`` that are legal for a (K, N) layer."""
    out = []
    for bn, bk, sk, nw, ns in itertools.product(space["BLOCK_N"], space["BLOCK_K"], space["SPLIT_K"],
                                                space["num_warps"], space["num_stages"]):
        if group_size % bk or k % bk or bn % 8 or sk > k // bk:
            continue
        out.append({"BLOCK_M": block_m, "BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": sk, "num_warps": nw, "num_stages": ns})
    return out


def attention_candidates(space: dict) -> list[dict]:
    return [{"block_n": bn, "num_warps": nw, "num_stages": ns, "num_splits": sp}
            for bn, nw, ns, sp in itertools.product(space["block_n"], space["num_warps"], space["num_stages"],
                                                    space["num_splits"])]


def n_copies(bytes_per_call: int, l2_bytes: int | None, *, floor_bytes: int = 128 << 20, cap: int = 512,
             budget_bytes: int | None = None) -> int:
    """How many distinct copies to cycle through so one pass streams well past the L2 cache."""
    target = max(floor_bytes, 4 * (l2_bytes or (32 << 20)))
    copies = max(2, math.ceil(target / max(1, bytes_per_call)))
    if budget_bytes is not None:
        copies = min(copies, max(2, budget_bytes // max(1, bytes_per_call)))
    return min(copies, cap)


def pick_best(results: list[dict], default: dict | None, min_gain: float) -> dict:
    """``results``: dicts with ``cfg``, ``us`` (microseconds per call) and ``valid``.

    Returns ``{"cfg", "us", "picked", "speedup", ...}``. A tuned config is chosen only if it is valid and at
    least ``min_gain`` faster than the default; otherwise the default is kept (``picked == "default"``).
    """
    valid = [r for r in results if r.get("valid") and r.get("us") is not None]
    if default is None or default.get("us") is None:
        raise ValueError("the default configuration could not be measured, so nothing can be compared against it")
    best = min(valid, key=lambda r: r["us"]) if valid else None
    if best is not None and best["us"] < default["us"] * (1.0 - min_gain):
        return {"cfg": best["cfg"], "us": best["us"], "picked": "tuned", "speedup": default["us"] / best["us"],
                "default_us": default["us"]}
    return {"cfg": default["cfg"], "us": default["us"], "picked": "default", "speedup": 1.0,
            "default_us": default["us"],
            "best_seen_us": best["us"] if best else None}


# --------------------------------------------------------------------------- #
# Measurement                                                                  #
# --------------------------------------------------------------------------- #
class Timer:
    """Microseconds per call of ``fns`` (a list of zero-argument callables), replayed as one CUDA graph."""

    def __init__(self, torch, use_graph: bool, replays: int):
        self.torch, self.use_graph, self.replays = torch, use_graph, replays

    def run(self, fns) -> float:
        torch = self.torch
        calls = len(fns)
        if torch.cuda.is_available() and not os.environ.get("TRITON_INTERPRET"):
            if self.use_graph:
                for fn in fns[:2]:  # compile / initialise outside the capture
                    fn()
                torch.cuda.synchronize()
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for fn in fns:
                        fn()
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                    for fn in fns:
                        fn()
                for _ in range(3):
                    graph.replay()
                torch.cuda.synchronize()
                step = graph.replay
            else:
                def step():
                    for fn in fns:
                        fn()
                step()
                torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(self.replays):
                step()
            end.record()
            torch.cuda.synchronize()
            us = start.elapsed_time(end) * 1e3 / (self.replays * calls)
            step = graph = None  # free the graph's private memory pool before the next candidate
            torch.cuda.empty_cache()
            return us
        t0 = time.perf_counter()  # CPU / interpreter smoke mode: wall time, meaningless as a benchmark
        for _ in range(self.replays):
            for fn in fns:
                fn()
        return (time.perf_counter() - t0) * 1e6 / (self.replays * calls)


def _l2_bytes(torch, device) -> int | None:
    if device.type != "cuda":
        return None
    return getattr(torch.cuda.get_device_properties(device), "L2_cache_size", None) or None


def _free_budget(torch, device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.mem_get_info(device)[0] * 0.4)


def tune_gemm_shape(torch, device, timer, k, n, group_size, space, log, *, l2_bytes, budget):
    from engine.kernels.dequant_gemm import awq_gemm, default_config

    g = torch.Generator().manual_seed(k * 31 + n)
    qw = torch.randint(-2**31, 2**31 - 1, (k, n // 8), dtype=torch.int32, generator=g)
    qz = torch.randint(-2**31, 2**31 - 1, (k // group_size, n // 8), dtype=torch.int32, generator=g)
    sc = (torch.rand(k // group_size, n, generator=g) * 0.01 + 0.002).half()
    x = torch.randn(1, k, generator=g).half()
    weight_bytes = qw.numel() * 4 + qz.numel() * 4 + sc.numel() * 2
    copies = n_copies(weight_bytes, l2_bytes, budget_bytes=budget, cap=COPY_CAP[0])
    ws = [(qw.clone().to(device), qz.clone().to(device), sc.clone().to(device)) for _ in range(copies)]
    x = x.to(device)

    ref = awq_gemm(x, *ws[0], group_size=group_size, out_dtype=torch.float32,
                   config=default_config(1, n, k, group_size, device)).float()
    # The baseline itself must agree with the independent dequantize-then-matmul reference, or every
    # comparison below would be against something wrong.
    from engine.kernels.dequant_gemm import awq_gemm_reference
    indep = awq_gemm_reference(x.cpu(), qw, qz, sc, None, group_size).float()
    base_diff = (ref.cpu() - indep).abs().max().item()
    if base_diff > 1e-3:
        raise RuntimeError(f"default awq_gemm disagrees with the reference by {base_diff:.3g} at K={k}, N={n}")
    default = {"cfg": default_config(1, n, k, group_size, device)}
    rows, measured = [], []
    candidates = gemm_candidates(k, n, group_size, space)
    for i, cfg in enumerate([{**default["cfg"]}] + candidates):
        is_default = i == 0
        row = {"K": k, "N": n, **{key: cfg.get(key, "") for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K", "SPLIT_K", "num_warps", "num_stages")},
               "is_default": is_default, "us": None, "GBs": None, "max_abs_diff": None, "valid": False, "error": ""}
        try:
            y = awq_gemm(x, *ws[0], group_size=group_size, out_dtype=torch.float32, config=cfg).float()
            diff = (y - ref).abs().max().item()
            row["max_abs_diff"] = diff
            row["valid"] = bool(diff <= 1e-3)
            if row["valid"]:
                fns = [(lambda w=w: awq_gemm(x, *w, group_size=group_size, config=cfg)) for w in ws]
                row["us"] = timer.run(fns)
                row["GBs"] = weight_bytes / (row["us"] * 1e-6) / 1e9
        except Exception as e:  # noqa: BLE001 - an illegal config (e.g. too much shared memory) is data, not a crash
            row["error"] = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"[:160]
        rows.append(row)
        if is_default:
            default["us"] = row["us"]
        else:
            measured.append({"cfg": cfg, "us": row["us"], "valid": row["valid"]})
        log(f"  gemm {k}x{n} [{i}/{len(candidates)}] {'DEFAULT' if is_default else ''} "
            + " ".join(f"{a}={b}" for a, b in cfg.items())
            + (f" -> {row['us']:.1f} us ({row['GBs']:.0f} GB/s)" if row["us"] else f" -> skipped {row['error'] or 'diff ' + str(row['max_abs_diff'])}"))
    choice = pick_best(measured, default, MIN_GAIN[0])
    choice.update({"K": k, "N": n, "weight_bytes": weight_bytes, "copies": copies,
                   "default_GBs": weight_bytes / (choice["default_us"] * 1e-6) / 1e9,
                   "chosen_GBs": weight_bytes / (choice["us"] * 1e-6) / 1e9})
    return choice, rows


def tune_attention(torch, device, timer, seq_lens, max_seq, space, log, *, l2_bytes, budget, hq=16, hkv=2, d=128):
    from engine.kernels.fused_attention import decode_attention_reference, fused_decode_attention

    g = torch.Generator().manual_seed(7)
    q = (torch.randn(hq, d, generator=g) * 2).half().to(device)
    cache_bytes = 2 * hkv * max_seq * d * 2
    copies = n_copies(cache_bytes, l2_bytes, budget_bytes=budget, cap=COPY_CAP[0])
    kv = [(torch.randn(hkv, max_seq, d, generator=g).half().to(device), torch.randn(hkv, max_seq, d, generator=g).half().to(device))
          for _ in range(copies)]
    refs = {s: decode_attention_reference(q.cpu(), kv[0][0].cpu(), kv[0][1].cpu(), s) for s in seq_lens}
    seqlen = torch.zeros(1, dtype=torch.int32, device=device)
    candidates = [{}] + attention_candidates(space)  # {} = the untouched defaults
    rows, per_cfg = [], []
    for i, cfg in enumerate(candidates):
        times, valid, worst, err = [], True, 0.0, ""
        try:
            for s in seq_lens:
                seqlen.fill_(s)  # the graph path reads the length from device memory; so does this timing
                out = fused_decode_attention(q, kv[0][0], kv[0][1], seqlen, **cfg).cpu().float()
                diff = (out - refs[s]).abs().max().item()
                worst = max(worst, diff)
                if diff > 1e-3:
                    valid = False
                    break
                fns = [(lambda kc=kc, vc=vc: fused_decode_attention(q, kc, vc, seqlen, **cfg)) for kc, vc in kv]
                times.append(timer.run(fns))
        except Exception as e:  # noqa: BLE001
            valid, err = False, f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"[:160]
        mean_us = sum(times) / len(times) if valid and times else None
        rows.append({**{"is_default": not cfg}, **(cfg or {"block_n": 64, "num_warps": "", "num_stages": "", "num_splits": 1}),
                     "seq_lens": "/".join(map(str, seq_lens)), "us_per_seq": "/".join(f"{t:.2f}" for t in times),
                     "mean_us": mean_us, "max_abs_diff": worst, "valid": valid, "error": err})
        per_cfg.append({"cfg": cfg, "us": mean_us, "valid": valid})
        log(f"  attention [{i}/{len(candidates) - 1}] {'DEFAULT' if not cfg else ''} {cfg} -> "
            + (f"{mean_us:.2f} us mean over S={seq_lens}" if mean_us else f"skipped {err or 'diff ' + str(worst)}"))
    default = per_cfg[0]
    default["cfg"] = {}
    choice = pick_best(per_cfg[1:], default, MIN_GAIN[0])
    choice.update({"kv_bytes_per_call_at_max_seq": cache_bytes, "copies": copies, "seq_lens": list(seq_lens)})
    return choice, rows


MIN_GAIN = [0.03]  # set from --min-gain (module-level so the sweep helpers need no extra plumbing)
COPY_CAP = [512]   # most distinct weight / KV copies to cycle through (smoke mode lowers it)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "tuning"))
    ap.add_argument("--space", choices=["quick", "standard", "full"], default="standard",
                    help="size of the sweep: at most 30 (quick), 180 (standard) or 405 (full) GEMM candidates per shape, before "
                         "illegal ones are dropped; each candidate is a separate Triton compile, so the sweep takes a while")
    ap.add_argument("--min-gain", type=float, default=0.03, help="a tuned config must beat the default by this fraction")
    ap.add_argument("--replays", type=int, default=10, help="graph replays per timing")
    ap.add_argument("--no-graph", action="store_true", help="time back-to-back eager calls instead of a CUDA graph (launch-bound!)")
    ap.add_argument("--attn-seq-lens", default="128,256,512", help="decode contexts the attention config is averaged over")
    ap.add_argument("--max-seq", type=int, default=1024, help="KV cache capacity used for the attention sweep (engine default 1024)")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run through the Triton interpreter; exercises the script, timings are meaningless")
    args = ap.parse_args(argv)
    MIN_GAIN[0] = args.min_gain
    COPY_CAP[0] = 2 if args.smoke else 512

    import torch

    if args.smoke:
        os.environ["TRITON_INTERPRET"] = "1"
    elif not torch.cuda.is_available():
        print("error: no CUDA GPU found. This tuner measures a real GPU; use --smoke to test the script on CPU.", file=sys.stderr)
        return 2
    device = torch.device("cpu" if args.smoke else "cuda")

    from engine.kernels.tuning import TunedConfigs
    from engine.model import QWEN2_5_3B

    cfg = QWEN2_5_3B
    kv = cfg.num_kv_heads * cfg.head_dim
    if args.smoke:
        shapes, gemm_space, attn_space = [(256, 256), (512, 128)], GEMM_SPACES["smoke"], ATTN_SPACES["smoke"]
        seq_lens, max_seq, replays = [16, 40], 64, 1
    else:
        shapes = sorted({(cfg.hidden_size, cfg.hidden_size), (cfg.hidden_size, kv),
                         (cfg.hidden_size, cfg.intermediate_size), (cfg.intermediate_size, cfg.hidden_size)})
        gemm_space, attn_space = GEMM_SPACES[args.space], ATTN_SPACES[args.space]
        seq_lens, max_seq, replays = [int(s) for s in args.attn_seq_lens.split(",")], args.max_seq, args.replays

    timer = Timer(torch, use_graph=not args.no_graph, replays=replays)
    log = lambda s: print(s, flush=True)  # noqa: E731
    l2, budget = _l2_bytes(torch, device), _free_budget(torch, device)
    out_dir = Path(args.out_dir)
    t0 = time.time()
    log(f"device: {torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu (smoke)'}; L2 {l2} B; "
        f"timing: {'CUDA graph replay' if (device.type == 'cuda' and not args.no_graph) else 'eager/wall'}; space={'smoke' if args.smoke else args.space}")

    gemm_best, gemm_rows, gemm_summary = {}, [], []
    for k, n in shapes:
        log(f"== awq_gemm K={k} N={n}")
        choice, rows = tune_gemm_shape(torch, device, timer, k, n, cfg.group_size, gemm_space, log, l2_bytes=l2, budget=budget)
        gemm_best[(k, n)] = choice["cfg"]
        gemm_rows += rows
        gemm_summary.append(choice)
    log(f"== attention (Hq={cfg.num_heads}, Hkv={cfg.num_kv_heads}, D={cfg.head_dim}) S={seq_lens}")
    attn_choice, attn_rows = tune_attention(torch, device, timer, seq_lens, max_seq, attn_space, log, l2_bytes=l2, budget=budget)

    meta = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu (smoke test - NOT a measurement)",
        "sm_count": torch.cuda.get_device_properties(device).multi_processor_count if device.type == "cuda" else None,
        "l2_cache_bytes": l2, "torch": torch.__version__, "python": platform.python_version(),
        "timing": ("CUDA graph replay of calls cycling through distinct weight/KV copies (working set >= 4x L2, >= 128 MiB)"
                   if device.type == "cuda" and not args.no_graph else "NOT GRAPH TIMED - launch overhead is included"),
        "min_gain": args.min_gain, "space": "smoke" if args.smoke else args.space,
        "checked_against": "fp32 reference, atol 1e-3, for every candidate",
        "awq_gemm_results": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k != "cfg"} | {"cfg": s["cfg"]} for s in gemm_summary],
        "attention_result": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in attn_choice.items()},
    }
    try:
        import triton
        meta["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        pass
    tuned = TunedConfigs(gemm=gemm_best, attention=attn_choice["cfg"], meta=meta)
    TunedConfigs.from_dict(tuned.to_dict(), group_size=cfg.group_size)  # the file we write must load
    tuned.save(out_dir / "best_configs.json")
    write_csv(out_dir / "awq_gemm_sweep.csv", gemm_rows)
    write_csv(out_dir / "attention_sweep.csv", attn_rows)

    log("\n== summary (microseconds per call; GB/s = packed weight bytes / time)")
    log(f"{'shape':>12} {'default us':>11} {'default GB/s':>13} {'chosen us':>10} {'chosen GB/s':>12} {'speedup':>8}  pick")
    for s in gemm_summary:
        log(f"{s['K']:>5}x{s['N']:<6} {s['default_us']:>11.1f} {s['default_GBs']:>13.1f} {s['us']:>10.1f} {s['chosen_GBs']:>12.1f} {s['speedup']:>7.2f}x  {s['picked']}")
    a = attn_choice
    log(f"{'attention':>12} {a['default_us']:>11.2f} {'':>13} {a['us']:>10.2f} {'':>12} {a['speedup']:>7.2f}x  {a['picked']} {a['cfg'] or '(defaults)'}")
    log(f"\nwrote {out_dir / 'best_configs.json'} (+ sweeps) in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
