"""Benchmark harness: decode tokens/s, TTFT, p99 inter-token latency, peak VRAM.

Every backend (this engine, llama.cpp via llama-cpp-python, vLLM's OpenAI server)
is driven through the same ``measure`` path with the same prompt token ids, greedy
decoding, EOS ignored, and a fixed number of new tokens. See SPEC.md section 5 for
the metric definitions.

This module imports only the standard library at module level so that it can run
inside the separate baseline virtualenvs (which do not have torch / this package
installed); heavy imports happen inside the backend that needs them.

    python -m engine.bench prompt --model Qwen/Qwen2.5-3B-Instruct-AWQ --length 128 --out results/prompt.json
    python -m engine.bench run --backend engine --prompt-file results/prompt.json --out results/benchmark.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime
import http.client
import json
import math
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

CSV_COLUMNS = [
    "timestamp_utc", "backend", "model", "quant", "device", "prompt_tokens", "new_tokens",
    "runs", "warmup", "decode_tok_s", "ttft_ms_p50", "itl_ms_p50", "itl_ms_p99",
    "itl_ms_max", "peak_vram_mib", "vram_baseline_mib", "vram_method", "notes",
]

BENCH_TEXT = (
    "Large language model inference has two phases. During prefill the whole prompt is "
    "processed in parallel and the work is dominated by matrix multiplications. During "
    "decode each new token requires reading every weight of the model from memory once, "
    "so throughput is limited by memory bandwidth rather than by arithmetic. "
)


# --------------------------------------------------------------------------- #
# Statistics                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class RunTiming:
    ttft_s: float
    itl_s: list[float]  # one entry per generated token after the first
    n_prompt: int = 0
    notes: list[str] = field(default_factory=list)


def percentile(values: Sequence[float], q: float) -> float:
    """``q``-th percentile with linear interpolation (numpy's default method)."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    xs = sorted(values)
    k = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(runs: Sequence[RunTiming]) -> dict:
    itl = [x for r in runs for x in r.itl_s]
    if not itl:
        raise ValueError("need at least 2 generated tokens per run to measure decode speed")
    return {
        "decode_tok_s": len(itl) / sum(itl),
        "ttft_ms_p50": percentile([r.ttft_s for r in runs], 50) * 1e3,
        "itl_ms_p50": percentile(itl, 50) * 1e3,
        "itl_ms_p99": percentile(itl, 99) * 1e3,
        "itl_ms_max": max(itl) * 1e3,
    }


# --------------------------------------------------------------------------- #
# VRAM                                                                        #
# --------------------------------------------------------------------------- #
def nvml_reader(index: int = 0) -> tuple[Callable[[], int], str]:
    """Return (function giving device-level used bytes, device name)."""
    import pynvml  # provided by the nvidia-ml-py package

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    name = pynvml.nvmlDeviceGetName(handle)
    name = name.decode() if isinstance(name, bytes) else name
    return (lambda: int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)), name


class VramMonitor:
    """Samples device-level used memory in a thread; reports the peak above the baseline."""

    def __init__(self, read_used_bytes: Callable[[], int] | None, interval_s: float = 0.005):
        self.read = read_used_bytes
        self.interval_s = interval_s
        self.baseline = 0
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self.read())
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        if self.read is None:
            return
        self.baseline = self.peak = self.read()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict | None:
        if self.read is None:
            return None
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak = max(self.peak, self.read())
        mib = 1024 * 1024
        return {"baseline_mib": self.baseline / mib, "peak_delta_mib": (self.peak - self.baseline) / mib}


# --------------------------------------------------------------------------- #
# Measurement                                                                 #
# --------------------------------------------------------------------------- #
class Backend(Protocol):
    name: str
    model: str
    quant: str

    def load(self) -> None: ...
    def generate(self, prompt_ids: Sequence[int], n_new: int) -> RunTiming: ...
    def close(self) -> None: ...


def measure(backend: Backend, prompt_ids: Sequence[int], n_new: int, *, warmup: int, runs: int,
            monitor: VramMonitor | None = None, device: str = "unknown",
            vram_method: str = "nvml-device-level") -> dict:
    if n_new < 2:
        raise ValueError("n_new must be >= 2")
    monitor = monitor or VramMonitor(None)
    monitor.start()
    try:
        backend.load()
        for _ in range(warmup):
            backend.generate(prompt_ids, n_new)
        timings = [backend.generate(prompt_ids, n_new) for _ in range(runs)]
    finally:
        vram = monitor.stop()
        backend.close()
    notes = sorted({n for t in timings for n in t.notes} | set(getattr(backend, "notes", [])))
    row = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "backend": backend.name, "model": backend.model, "quant": backend.quant, "device": device,
        "prompt_tokens": len(prompt_ids), "new_tokens": n_new, "runs": runs, "warmup": warmup,
        **summarize(timings),
        "peak_vram_mib": "" if vram is None else vram["peak_delta_mib"],
        "vram_baseline_mib": "" if vram is None else vram["baseline_mib"],
        "vram_method": vram_method if vram is not None else "unavailable",
        "notes": "; ".join(notes),
    }
    return row


def append_csv(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})


# --------------------------------------------------------------------------- #
# Backends                                                                    #
# --------------------------------------------------------------------------- #
ENGINE_BACKENDS = ("engine", "engine-graph", "engine-tuned", "engine-graph-tuned", "engine-reference-ops")


class EngineBackend:
    """This engine. ``reference_ops=True`` swaps every custom op for its PyTorch
    reference (profiling baseline; materializes one weight matrix at a time).

    ``cuda_graph=True`` decodes through a ``DeviceDecoder`` that replays a captured CUDA graph;
    ``tuned_config`` is a file written by ``scripts/tune_kernels.py``. Both are recorded in the CSV
    ``notes`` column, and if the graph cannot be captured the run falls back to the eager loop and the
    note says so (``cuda_graph=FALLBACK ...``) - it is never reported as a graph run."""

    quant = "AWQ int4 g128"

    def __init__(self, model: str, *, loader: str = "autoawq", max_seq_len: int = 1024,
                 reference_ops: bool = False, prebuilt=None, cuda_graph: bool = False,
                 tuned_config: str | None = None, name: str | None = None):
        self.model = model
        self.loader = loader
        self.max_seq_len = max_seq_len
        self.reference_ops = reference_ops
        self.cuda_graph = cuda_graph
        self.tuned_config = tuned_config
        self.name = name or ("engine-reference-ops" if reference_ops else "engine")
        self.notes = [f"loader={loader}"]
        if tuned_config:
            self.notes.append(f"tuned_config={tuned_config}")
        self._m = prebuilt
        self._cache = None
        self._decoder = None

    def load(self) -> None:
        from .generate import KVCache
        from .graph import DeviceDecoder
        from .model import EngineOptions, load_awq_model

        if self._m is None:
            opts = (EngineOptions("reference", "reference", "reference") if self.reference_ops
                    else EngineOptions(tuned_config=self.tuned_config))
            self._m = load_awq_model(self.model, loader=self.loader, device="cuda",
                                     max_seq_len=self.max_seq_len, options=opts)
        self._cache = KVCache(self._m.cfg, self._m.max_seq_len, self._m.device)
        if self.cuda_graph:
            self._decoder = DeviceDecoder(self._m, self._cache, use_graph=True)

    def generate(self, prompt_ids, n_new) -> RunTiming:
        from .generate import generate

        r = generate(self._m, prompt_ids, n_new, cache=self._cache, ignore_eos=True, decoder=self._decoder)
        return RunTiming(r.ttft_s, r.itl_s, r.prompt_tokens)

    def close(self) -> None:
        if self._decoder is not None:  # the mode is only known once the first (warm-up) run has tried to capture
            self.notes.append(self._decoder.describe())
        self._m = self._cache = self._decoder = None


class LlamaCppBackend:
    """llama.cpp through llama-cpp-python's token generator (one token per iteration)."""

    quant = "GGUF Q4_K_M"

    def __init__(self, gguf_path: str, *, n_ctx: int = 1024, flash_attn: bool = True,
                 check_text: str | None = None, check_ids: Sequence[int] | None = None):
        self.name = "llama.cpp"
        self.model = Path(gguf_path).name
        self.gguf_path, self.n_ctx, self.flash_attn = gguf_path, n_ctx, flash_attn
        self.notes = [f"flash_attn={flash_attn}", "n_gpu_layers=-1"]
        self._check_text, self._check_ids = check_text, check_ids
        self._llm = None

    def load(self) -> None:
        from llama_cpp import Llama

        self._llm = Llama(model_path=self.gguf_path, n_gpu_layers=-1, n_ctx=self.n_ctx,
                          flash_attn=self.flash_attn, verbose=False)
        if self._check_text is not None and self._check_ids is not None:
            same = self._llm.tokenize(self._check_text.encode("utf-8"), add_bos=False) == list(self._check_ids)
            self.notes.append("tokenizer_agrees_with_hf=yes" if same else "tokenizer_agrees_with_hf=NO")

    def generate(self, prompt_ids, n_new) -> RunTiming:
        llm = self._llm
        llm.reset()  # otherwise generate() reuses the cached prompt prefix and skips prefill
        stamps: list[float] = []
        t0 = time.perf_counter()
        for _ in llm.generate(list(prompt_ids), top_k=1, top_p=1.0, min_p=0.0, temp=0.0,
                              repeat_penalty=1.0, reset=True):
            stamps.append(time.perf_counter())
            if len(stamps) >= n_new:
                break
        return RunTiming(stamps[0] - t0, [b - a for a, b in zip(stamps, stamps[1:])], len(prompt_ids))

    def close(self) -> None:
        if self._llm is not None:
            self._llm.close()
        self._llm = None


def sse_events_to_timing(t0: float, events: Sequence[tuple[float, int]], n_prompt: int) -> RunTiming:
    """``events``: (arrival time, tokens in that chunk). Tokens that arrive in one chunk
    share the chunk's interval equally; extra tokens in the *first* chunk get zero."""
    if not events:
        raise ValueError("no tokens received")
    notes = []
    itl: list[float] = [0.0] * (events[0][1] - 1)
    prev = events[0][0]
    for t, k in events[1:]:
        itl.extend([(t - prev) / k] * k)
        prev = t
    if any(k > 1 for _, k in events):
        notes.append("some stream chunks carried >1 token")
    return RunTiming(events[0][0] - t0, itl, n_prompt, notes)


def stream_completion(host: str, port: int, body: dict, timeout_s: float = 600) -> tuple[float, list[tuple[float, int]]]:
    """POST an OpenAI-style streaming completion; return (t0, [(arrival, n_tokens), ...]).

    Requires each chunk to carry ``choices[0].token_ids`` (vLLM: ``return_token_ids``).
    """
    conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        t0 = time.perf_counter()
        conn.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:300]!r}")
        events: list[tuple[float, int]] = []
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            now = time.perf_counter()
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            ids = json.loads(payload)["choices"][0].get("token_ids") or []
            if ids:
                events.append((now, len(ids)))
        return t0, events
    finally:
        conn.close()


class VllmServerBackend:
    """vLLM's OpenAI-compatible server (``vllm serve``), streamed over HTTP.

    The HTTP hop is included in the measured latencies, and VRAM includes whatever
    KV-cache pre-allocation ``--gpu-memory-utilization`` asks for.
    """

    quant = "AWQ int4 g128"

    def __init__(self, model: str, *, vllm_bin: str = "vllm", max_model_len: int = 1024,
                 gpu_memory_utilization: float = 0.6, port: int = 8765, log_path: str | None = None,
                 startup_timeout_s: float = 900):
        self.name = "vllm"
        self.model = model
        self.vllm_bin, self.max_model_len = vllm_bin, max_model_len
        self.gpu_memory_utilization, self.port = gpu_memory_utilization, port
        self.log_path, self.startup_timeout_s = log_path, startup_timeout_s
        self.notes = [f"gpu_memory_utilization={gpu_memory_utilization}", "max_num_seqs=1",
                      "transport=http-sse"]
        self._proc: subprocess.Popen | None = None

    def load(self) -> None:
        cmd = [self.vllm_bin, "serve", self.model, "--quantization", "awq", "--dtype", "float16",
               "--max-model-len", str(self.max_model_len), "--max-num-seqs", "1",
               "--gpu-memory-utilization", str(self.gpu_memory_utilization),
               "--port", str(self.port), "--served-model-name", "bench"]
        log = open(self.log_path, "wb") if self.log_path else subprocess.DEVNULL
        self._proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.time() + self.startup_timeout_s
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"vllm exited with code {self._proc.returncode}; see {self.log_path}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=2) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(1.0)
        raise TimeoutError("vllm server did not become ready")

    def generate(self, prompt_ids, n_new) -> RunTiming:
        body = {"model": "bench", "prompt": list(prompt_ids), "max_tokens": n_new, "temperature": 0.0,
                "stream": True, "ignore_eos": True, "return_token_ids": True, "stream_interval": 1}
        t0, events = stream_completion("127.0.0.1", self.port, body)
        timing = sse_events_to_timing(t0, events, len(prompt_ids))
        got = sum(k for _, k in events)
        if got != n_new:
            timing.notes.append(f"received {got} tokens, expected {n_new}")
        return timing

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None


# --------------------------------------------------------------------------- #
# Prompt construction and CLI                                                 #
# --------------------------------------------------------------------------- #
def fit_to_length(ids: Sequence[int], length: int) -> list[int]:
    """Repeat / truncate ``ids`` to exactly ``length`` tokens."""
    if not ids:
        raise ValueError("empty token list")
    reps = -(-length // len(ids))
    return (list(ids) * reps)[:length]


def make_prompt(model: str, length: int, out: str | Path) -> dict:
    from transformers import AutoTokenizer

    from .model import resolve_model_dir

    tok = AutoTokenizer.from_pretrained(str(resolve_model_dir(model)))
    base_ids = tok(BENCH_TEXT, add_special_tokens=False).input_ids
    ids = fit_to_length(base_ids, length)
    # base_text/base_ids let a backend check that its tokenizer agrees with ours on clean text
    # (the repeated, truncated prompt itself need not survive a decode->encode round trip).
    payload = {"model": model, "length": length, "ids": ids, "base_text": BENCH_TEXT, "base_ids": base_ids}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload))
    return payload


def _build_backend(args, prompt: dict) -> Backend:
    if args.backend in ENGINE_BACKENDS:
        tuned = "tuned" in args.backend
        if tuned and not args.tuned_config:
            raise SystemExit(f"--tuned-config is required for --backend {args.backend} "
                             "(write one with scripts/tune_kernels.py on the GPU)")
        return EngineBackend(args.model, loader=args.loader, max_seq_len=args.max_seq_len,
                             reference_ops=args.backend == "engine-reference-ops",
                             cuda_graph="graph" in args.backend,
                             tuned_config=args.tuned_config if tuned else None, name=args.backend)
    if args.backend == "llamacpp":
        if not args.gguf:
            raise SystemExit("--gguf is required for the llamacpp backend")
        return LlamaCppBackend(args.gguf, n_ctx=args.max_seq_len, flash_attn=not args.no_flash_attn,
                               check_text=prompt.get("base_text"), check_ids=prompt.get("base_ids"))
    if args.backend == "vllm":
        return VllmServerBackend(args.model, vllm_bin=args.vllm_bin, max_model_len=args.max_seq_len,
                                 gpu_memory_utilization=args.gpu_memory_utilization, port=args.port,
                                 log_path=args.log)
    raise SystemExit(f"unknown backend {args.backend}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="engine.bench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prompt", help="write the shared prompt token ids")
    p.add_argument("--model", required=True)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--out", required=True)

    r = sub.add_parser("run", help="benchmark one backend and append a CSV row")
    r.add_argument("--backend", required=True, choices=[*ENGINE_BACKENDS, "llamacpp", "vllm"])
    r.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct-AWQ")
    r.add_argument("--loader", default="autoawq", choices=["autoawq", "safetensors"])
    r.add_argument("--gguf", help="path to the GGUF file (llamacpp backend)")
    r.add_argument("--tuned-config", help="file written by scripts/tune_kernels.py (the *-tuned engine backends)")
    r.add_argument("--prompt-file", required=True)
    r.add_argument("--new-tokens", type=int, default=128)
    r.add_argument("--runs", type=int, default=10)
    r.add_argument("--warmup", type=int, default=2)
    r.add_argument("--max-seq-len", type=int, default=1024)
    r.add_argument("--out", required=True)
    r.add_argument("--no-flash-attn", action="store_true")
    r.add_argument("--vllm-bin", default="vllm")
    r.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    r.add_argument("--port", type=int, default=8765)
    r.add_argument("--log")
    args = ap.parse_args(argv)

    if args.cmd == "prompt":
        payload = make_prompt(args.model, args.length, args.out)
        print(f"wrote {len(payload['ids'])} prompt tokens to {args.out}")
        return 0

    prompt = json.loads(Path(args.prompt_file).read_text())
    if len(prompt["ids"]) + args.new_tokens > args.max_seq_len:
        raise SystemExit("prompt + new tokens exceed --max-seq-len")
    try:
        read, device = nvml_reader()
        monitor, method = VramMonitor(read), "nvml-device-level"
    except Exception as e:  # no NVIDIA driver / pynvml: report VRAM as unavailable, never invent it
        print(f"warning: NVML unavailable ({e}); VRAM will be blank", file=sys.stderr)
        monitor, method, device = VramMonitor(None), "unavailable", "unknown"
    row = measure(_build_backend(args, prompt), prompt["ids"], args.new_tokens, warmup=args.warmup,
                  runs=args.runs, monitor=monitor, device=device, vram_method=method)
    append_csv(args.out, row)
    if "cuda_graph=FALLBACK" in row["notes"]:
        print("WARNING: the CUDA graph could not be used; this row measures the EAGER loop. "
              f"Reason in the notes column: {row['notes']}", file=sys.stderr)
    print(json.dumps({k: row[k] for k in ("backend", "decode_tok_s", "ttft_ms_p50", "itl_ms_p99", "peak_vram_mib")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
