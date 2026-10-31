"""Test-matrix item 5: the benchmark harness produces tokens/s, TTFT, p99 and VRAM.

These tests prove the harness arithmetic and plumbing with fake backends / a local
fake SSE server. They do not (and cannot) produce real GPU numbers; those come from
``make benchmark`` on the target machine.
"""

import csv
import http.server
import json
import threading
import time

import numpy as np
import pytest

from engine.bench import (
    CSV_COLUMNS,
    RunTiming,
    VramMonitor,
    append_csv,
    fit_to_length,
    measure,
    percentile,
    sse_events_to_timing,
    stream_completion,
    summarize,
)

MIB = 1024 * 1024


def test_percentile_matches_numpy_linear_interpolation():
    rng = np.random.default_rng(0)
    for n in (2, 3, 10, 101, 1000):
        xs = rng.random(n).tolist()
        for q in (0, 1, 50, 90, 99, 99.9, 100):
            assert percentile(xs, q) == pytest.approx(float(np.percentile(xs, q)), rel=1e-12, abs=1e-12)


def test_summarize_known_values():
    # 3 runs x 100 inter-token gaps of 10 ms, except one 50 ms outlier per run
    runs = [RunTiming(0.2, [0.010] * 99 + [0.050]) for _ in range(3)]
    s = summarize(runs)
    assert s["decode_tok_s"] == pytest.approx(300 / (3 * (99 * 0.010 + 0.050)))
    assert s["ttft_ms_p50"] == pytest.approx(200.0)
    assert s["itl_ms_p50"] == pytest.approx(10.0)
    assert s["itl_ms_max"] == pytest.approx(50.0)
    # 3 outliers among 300 samples: the 99th percentile interpolates just below 50 ms
    assert 10.0 < s["itl_ms_p99"] <= 50.0


def test_summarize_requires_decode_tokens():
    with pytest.raises(ValueError):
        summarize([RunTiming(0.1, [])])


def test_vram_monitor_reports_peak_above_baseline():
    values = iter([100 * MIB, 110 * MIB, 400 * MIB, 250 * MIB])
    last = [100 * MIB]

    def reader():
        try:
            last[0] = next(values)
        except StopIteration:
            pass
        return last[0]

    mon = VramMonitor(reader, interval_s=0.001)
    mon.start()
    time.sleep(0.05)
    out = mon.stop()
    assert out["baseline_mib"] == pytest.approx(100)
    assert out["peak_delta_mib"] == pytest.approx(300)


def test_vram_monitor_without_reader_is_inert():
    mon = VramMonitor(None)
    mon.start()
    assert mon.stop() is None


class FakeBackend:
    name, model, quant = "fake", "fake-model", "fake-quant"
    notes = ["fake-note"]

    def __init__(self):
        self.loaded = self.closed = False
        self.calls = 0

    def load(self):
        self.loaded = True

    def generate(self, prompt_ids, n_new):
        self.calls += 1
        return RunTiming(0.05, [0.02] * (n_new - 1), len(prompt_ids))

    def close(self):
        self.closed = True


def test_measure_returns_all_four_metrics_and_runs_warmup(tmp_path):
    backend = FakeBackend()
    ticks = iter(range(10**6))
    mon = VramMonitor(lambda: (1000 + next(ticks) % 3 * 100) * MIB, interval_s=0.001)
    row = measure(backend, list(range(16)), 33, warmup=2, runs=4, monitor=mon, device="fake-gpu")
    assert backend.loaded and backend.closed and backend.calls == 6
    assert row["decode_tok_s"] == pytest.approx(50.0)
    assert row["ttft_ms_p50"] == pytest.approx(50.0)
    assert row["itl_ms_p99"] == pytest.approx(20.0)
    assert row["peak_vram_mib"] > 0
    assert row["prompt_tokens"] == 16 and row["new_tokens"] == 33 and row["runs"] == 4
    assert "fake-note" in row["notes"]

    out = tmp_path / "b.csv"
    append_csv(out, row)
    append_csv(out, row)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2 and list(rows[0].keys()) == CSV_COLUMNS
    assert float(rows[0]["decode_tok_s"]) == pytest.approx(50.0, abs=1e-3)


def test_measure_leaves_vram_blank_when_unavailable():
    row = measure(FakeBackend(), [1, 2, 3], 5, warmup=0, runs=1, monitor=VramMonitor(None))
    assert row["peak_vram_mib"] == "" and row["vram_method"] == "unavailable"


def test_measure_rejects_single_token_runs():
    with pytest.raises(ValueError):
        measure(FakeBackend(), [1], 1, warmup=0, runs=1)


def test_engine_backend_runs_through_the_harness_on_a_synthetic_model():
    from engine.bench import EngineBackend
    from engine.model import ModelConfig, build_synthetic_model

    from .conftest import DEVICE

    cfg = ModelConfig(256, 512, 2, 4, 2, 512, 1e-6, 10000.0)
    model = build_synthetic_model(cfg, seed=11, device=DEVICE, max_seq_len=64)
    row = measure(EngineBackend("synthetic", prebuilt=model), [11, 42, 7, 300, 5], 6, warmup=1, runs=2)
    assert row["backend"] == "engine"
    assert row["decode_tok_s"] > 0 and row["ttft_ms_p50"] > 0 and row["itl_ms_p99"] > 0


def test_fit_to_length():
    assert fit_to_length([1, 2, 3], 7) == [1, 2, 3, 1, 2, 3, 1]
    assert fit_to_length([1, 2, 3], 2) == [1, 2]


# --------------------------------------------------------------------------- #
# Streaming (vLLM-style SSE) client against a local fake server                #
# --------------------------------------------------------------------------- #
class _SSEHandler(http.server.BaseHTTPRequestHandler):
    chunks = [1, 1, 2, 1]  # tokens per chunk
    delay = 0.03

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["stream"] is True and body["return_token_ids"] is True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        tok = 0
        for k in self.chunks:
            time.sleep(self.delay)
            ids = list(range(tok, tok + k))
            tok += k
            payload = {"choices": [{"index": 0, "text": "x", "token_ids": ids, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()
        final = {"choices": [{"index": 0, "text": "", "token_ids": [], "finish_reason": "length"}]}
        self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())
        self.wfile.flush()


@pytest.fixture()
def sse_server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _SSEHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv.server_address[1]
    srv.shutdown()


def test_stream_client_counts_tokens_and_times_chunks(sse_server):
    t0, events = stream_completion("127.0.0.1", sse_server, {"stream": True, "return_token_ids": True})
    assert [k for _, k in events] == [1, 1, 2, 1]  # the empty finish chunk is not a token
    gaps = [b[0] - a[0] for a, b in zip(events, events[1:])]
    assert all(g >= 0.02 for g in gaps)  # chunks were observed as they arrived, not batched
    timing = sse_events_to_timing(t0, events, n_prompt=8)
    assert timing.ttft_s >= 0.025
    assert len(timing.itl_s) == 4  # 5 tokens -> 4 inter-token gaps
    assert any("carried >1 token" in n for n in timing.notes)


def test_sse_events_to_timing_splits_multi_token_chunks():
    t = sse_events_to_timing(10.0, [(10.5, 1), (10.6, 1), (10.8, 2)], n_prompt=3)
    assert t.ttft_s == pytest.approx(0.5)
    assert t.itl_s == pytest.approx([0.1, 0.1, 0.1])
    with pytest.raises(ValueError):
        sse_events_to_timing(0.0, [], 1)


def test_plot_script_renders_from_a_benchmark_csv(tmp_path):
    """Pipeline check only: the rows below are synthetic test fixtures in a temp dir."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "plot_results", Path(__file__).resolve().parents[1] / "scripts" / "plot_results.py")
    plot_results = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot_results)

    csv_path, png = tmp_path / "b.csv", tmp_path / "t.png"
    for name, tok in (("engine", 50.0), ("llama.cpp", 40.0)):
        backend = FakeBackend()
        backend.name = name
        row = measure(backend, [1, 2, 3], 11, warmup=0, runs=1, device="fixture-gpu")
        row["decode_tok_s"] = tok
        append_csv(csv_path, row)
    assert plot_results.main(["--csv", str(csv_path), "--out", str(png)]) == 0
    assert png.stat().st_size > 10_000
    assert plot_results.main(["--csv", str(tmp_path / "missing.csv"), "--out", str(png)]) == 1
