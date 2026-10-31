"""The benchmark's engine variants: which flags each backend name turns on, and that a run whose CUDA graph
could not be captured is labelled as the eager run it actually was.
"""

import argparse
import json

import pytest

from engine import bench
from engine.model import ModelConfig, build_synthetic_model

from .conftest import DEVICE

TINY = ModelConfig(hidden_size=256, intermediate_size=512, num_layers=2, num_heads=4,
                   num_kv_heads=2, vocab_size=512, rms_norm_eps=1e-6, rope_theta=10000.0)


def _args(backend, tuned=None):
    return argparse.Namespace(backend=backend, model="m", loader="autoawq", max_seq_len=64, tuned_config=tuned)


@pytest.mark.parametrize("name,graph,tuned", [
    ("engine", False, False), ("engine-graph", True, False),
    ("engine-tuned", False, True), ("engine-graph-tuned", True, True),
])
def test_backend_names_select_graph_and_tuning(name, graph, tuned, tmp_path):
    cfg = str(tmp_path / "t.json") if tuned else None
    b = bench._build_backend(_args(name, cfg), {})
    assert (b.name, b.cuda_graph, b.tuned_config) == (name, graph, cfg)
    assert (f"tuned_config={cfg}" in b.notes) == tuned


def test_the_reference_ablation_never_gets_graph_or_tuning():
    b = bench._build_backend(_args("engine-reference-ops", "ignored.json"), {})
    assert b.reference_ops and not b.cuda_graph and b.tuned_config is None


@pytest.mark.parametrize("name", ["engine-tuned", "engine-graph-tuned"])
def test_a_tuned_backend_without_a_config_file_is_refused(name):
    with pytest.raises(SystemExit, match="--tuned-config"):
        bench._build_backend(_args(name), {})


def _model():
    return build_synthetic_model(TINY, seed=11, device=DEVICE, max_seq_len=64, embed_std=0.05)


def test_a_graph_run_that_cannot_capture_is_recorded_as_a_fallback_not_a_graph_run(monkeypatch):
    from engine.graph import DeviceDecoder

    def boom(self):
        raise RuntimeError("operation failed due to a previous error during capture")

    monkeypatch.setattr(DeviceDecoder, "_capture", boom)
    backend = bench.EngineBackend("m", prebuilt=_model(), cuda_graph=True, name="engine-graph")
    row = bench.measure(backend, [11, 42, 7, 300, 5, 9], 6, warmup=1, runs=2, device="test")
    assert row["backend"] == "engine-graph"
    assert "cuda_graph=FALLBACK to eager loop" in row["notes"] and "previous error during capture" in row["notes"]
    assert "cuda_graph=on" not in row["notes"]
    assert row["decode_tok_s"] > 0  # the eager loop still produced timings


def test_the_graph_backend_reports_when_the_graph_is_in_use(monkeypatch):
    from engine.graph import DeviceDecoder

    # Stand-in for a successful capture on a CPU machine: the decoder runs the same device-position step eagerly.
    orig = DeviceDecoder.ensure_ready

    def ready_as_device_eager(self):
        self.use_graph = False
        orig(self)

    monkeypatch.setattr(DeviceDecoder, "ensure_ready", ready_as_device_eager)
    backend = bench.EngineBackend("m", prebuilt=_model(), cuda_graph=True, name="engine-graph")
    row = bench.measure(backend, [11, 42, 7, 300, 5, 9], 6, warmup=1, runs=2, device="test")
    assert "cuda_graph=off(device-position eager)" in row["notes"]


def test_the_cli_warns_loudly_when_a_graph_row_is_really_eager(monkeypatch, tmp_path, capsys):
    from engine.graph import DeviceDecoder

    monkeypatch.setattr(DeviceDecoder, "_capture", lambda self: (_ for _ in ()).throw(RuntimeError("no capture")))
    monkeypatch.setattr(bench, "nvml_reader", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no NVML")))
    prompt = tmp_path / "p.json"
    prompt.write_text(json.dumps({"ids": [11, 42, 7, 300, 5, 9], "base_text": "x", "base_ids": [1]}))
    monkeypatch.setattr(bench, "_build_backend",
                        lambda a, p: bench.EngineBackend("m", prebuilt=_model(), cuda_graph=True, name="engine-graph"))
    out = tmp_path / "b.csv"
    rc = bench.main(["run", "--backend", "engine-graph", "--prompt-file", str(prompt), "--new-tokens", "4",
                     "--runs", "1", "--warmup", "1", "--max-seq-len", "64", "--out", str(out)])
    assert rc == 0
    assert "WARNING: the CUDA graph could not be used" in capsys.readouterr().err
    assert "cuda_graph=FALLBACK" in out.read_text()
