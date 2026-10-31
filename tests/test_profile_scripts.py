"""Parsing / aggregation logic of the profiling scripts, on synthetic inputs.

The ncu sample below follows the documented layout of `ncu --csv --page raw
--print-units base` (a header line, a units row, one row per kernel launch). It is a
FORMAT ASSUMPTION: it has not been checked against real Nsight Compute output, which is
unavailable here, so a mismatch on a GPU machine is possible; the parser fails loudly
with the first lines of the output in that case.
"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    import sys

    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


report = _load("profile_report")
decode = _load("profile_decode")

NCU_SAMPLE = '''==PROF== Connected to process 1234
"ID","Process ID","Process Name","Kernel Name","gpu__time_duration.sum","dram__bytes_read.sum","dram__bytes_write.sum","dram__throughput.avg.pct_of_peak_sustained_elapsed"
"","","","","nsecond","byte","byte","%"
"0","1234","python","_awq_gemm_kernel_0d1d2d","10,000","4,000,000","100","80.0"
"1","1234","python","_awq_gemm_kernel_0d1d2d","30000","12000000","300","40.0"
"2","1234","python","_decode_attn_kernel","5000","500000","0","20.0"
"3","1234","python","void at::native::elementwise_kernel<mul>","2000","400000","400000","10.0"
==PROF== Disconnected from process 1234
'''


def test_categorize_known_kernel_names():
    assert decode.categorize("_awq_gemm_kernel_0d1d2d") == "custom: fused dequant+GEMM"
    assert decode.categorize("_decode_attn_kernel") == "custom: fused decode attention"
    assert decode.categorize("ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn") == "GEMM / GEMV (cuBLAS, incl. lm_head)"
    assert decode.categorize("something_unrecognized") == "other"


def test_parse_ncu_raw_csv_and_aggregate():
    rows, units = report.parse_ncu_raw_csv(NCU_SAMPLE)
    assert len(rows) == 4 and units["gpu__time_duration.sum"] == "nsecond"
    agg = report.aggregate_ncu(rows, units, steps=2)
    g = agg["custom: fused dequant+GEMM"]
    assert g["launches_per_step"] == 1.0
    assert g["gpu_time_us_per_step"] == pytest.approx((10000 + 30000) / 1e3 / 2)
    assert g["dram_read_MB_per_step"] == pytest.approx((4e6 + 12e6) / 1e6 / 2)
    # time-weighted mean of 80% (10 us) and 40% (30 us) is 50%
    assert g["dram_throughput_pct_time_weighted"] == pytest.approx(50.0)
    t = report.totals(agg)
    assert t["gpu_time_us_per_step"] == pytest.approx((10000 + 30000 + 5000 + 2000) / 1e3 / 2)


def test_parse_ncu_fails_loudly_on_unexpected_output():
    with pytest.raises(ValueError, match="no CSV header"):
        report.parse_ncu_raw_csv("==ERROR== ERR_NVGPUCTRPERM - permission denied\n")


def test_unexpected_units_are_rejected():
    rows, units = report.parse_ncu_raw_csv(NCU_SAMPLE.replace('"nsecond"', '"usecond"'))
    with pytest.raises(ValueError, match="print-units base"):
        report.aggregate_ncu(rows, units, steps=1)


def test_kernels_from_chrome_trace_and_summary(tmp_path):
    trace = {"traceEvents": [
        {"ph": "X", "cat": "kernel", "name": "_awq_gemm_kernel", "ts": 1, "dur": 40.0},
        {"ph": "X", "cat": "kernel", "name": "_awq_gemm_kernel", "ts": 2, "dur": 60.0},
        {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 3, "dur": 999.0},
        {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy DtoH", "ts": 4, "dur": 5.0},
    ]}
    p = tmp_path / "t.json"
    p.write_text(json.dumps(trace))
    ks = decode.kernels_from_chrome_trace(p)
    assert [k["dur_us"] for k in ks] == [40.0, 60.0]
    rows = decode.summarize_kernels(ks, steps=2)
    assert rows[0]["calls"] == 2 and rows[0]["total_us"] == 100.0 and rows[0]["calls_per_step"] == 1.0


# --------------------------------------------------------------------------- #
# Regression: the WSL2 run where torch.profiler recorded no CUDA kernels         #
# (busy_ms == 0) crashed with ZeroDivisionError in profile_decode.py.            #
# --------------------------------------------------------------------------- #
def _summary(rows, timing=None, copy_bw=200.0):
    return decode.build_summary(variant="custom", steps=10, device="test-gpu", rows=rows, wall_s=0.1,
                                useful_bytes=2.0e9, copy_bw_gbs=copy_bw, timing=timing)


def test_build_summary_does_not_divide_by_zero_when_cupti_recorded_nothing():
    s = _summary(rows=[])  # exactly the failing situation: no kernel events at all
    assert s["gpu_busy_ms_per_step"] is None and s["kernel_launches_per_step"] is None
    assert s["useful_bandwidth_GBs"] is None and s["useful_bandwidth_pct_of_measured_copy_bw"] is None
    assert len(s["warnings"]) == 1 and "no CUDA kernel events" in s["warnings"][0]
    assert s["wall_ms_per_step_under_profiler"] == pytest.approx(100.0)  # CPU-side timing is still reported


def test_build_summary_with_kernel_events():
    rows = [{"total_us": 40_000.0, "calls": 100}, {"total_us": 60_000.0, "calls": 50}]  # 100 ms over 10 steps
    s = _summary(rows)
    assert s["gpu_busy_ms_per_step"] == pytest.approx(10.0)
    assert s["kernel_launches_per_step"] == pytest.approx(15.0)
    assert s["useful_bandwidth_GBs"] == pytest.approx(2.0e9 / 10e-3 / 1e9)  # 200 GB/s
    assert s["useful_bandwidth_pct_of_measured_copy_bw"] == pytest.approx(100.0)
    assert s["warnings"] == []


def test_build_summary_uses_cupti_free_timing_when_kernel_events_are_missing():
    s = _summary(rows=[], timing={"eager_ms_per_step": 80.0, "graph_replay_ms_per_step": 20.0})
    assert s["launch_overhead_fraction_of_eager"] == pytest.approx(0.75)
    assert s["useful_bandwidth_GBs_graph"] == pytest.approx(2.0e9 / 20e-3 / 1e9)  # 100 GB/s
    assert s["useful_bandwidth_pct_of_measured_copy_bw_graph"] == pytest.approx(50.0)
    assert s["useful_bandwidth_GBs"] is None  # still not invented from CUPTI data that does not exist


def test_build_summary_records_graph_capture_errors_instead_of_hiding_them():
    s = _summary(rows=[], timing={"eager_ms_per_step": 80.0, "graph_capture_error": "RuntimeError: boom"})
    assert s["graph_replay_ms_per_step"] is None and s["launch_overhead_fraction_of_eager"] is None
    assert s["timing_errors"] == {"graph_capture_error": "RuntimeError: boom"}


def test_measure_launch_vs_gpu_reports_instead_of_raising():
    import torch

    from .conftest import HAS_CUDA

    calls = []
    out = decode.measure_launch_vs_gpu(lambda: calls.append(1), lambda: None, steps=3)
    if HAS_CUDA:
        # on a GPU machine this really captures and replays a (trivial) CUDA graph
        x = torch.ones(8, device="cuda")
        out = decode.measure_launch_vs_gpu(lambda: x.add_(1), lambda: x.add_(1), steps=3)
        assert out["eager_ms_per_step"] > 0 and out["graph_replay_ms_per_step"] > 0
    else:
        assert "eager_error" in out and "graph_capture_error" in out  # no CUDA: errors are data, not exceptions


def test_report_survives_a_summary_without_kernel_data(tmp_path, capsys):
    (tmp_path / "profile_custom_summary.json").write_text(json.dumps(_summary(rows=[], timing={
        "eager_ms_per_step": 77.0, "graph_replay_ms_per_step": 12.0})))
    (tmp_path / "profile_custom_kernels.csv").write_text("name,category,calls,calls_per_step,total_us,avg_us\n")
    assert report.main(["--results", str(tmp_path)]) == 0
    out = capsys.readouterr()
    assert "n/a" in out.out and "CUDA-graph replay" in out.out and "WARNING" in out.out
    assert not (tmp_path / "dram_throughput.png").exists()  # nothing real to plot, so nothing is drawn
