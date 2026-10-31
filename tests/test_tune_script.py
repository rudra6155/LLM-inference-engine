"""scripts/tune_kernels.py: the selection logic, and the whole script end to end in --smoke mode.

The GPU timings the script exists to produce cannot be tested without a GPU; what can be tested - and is, here -
is everything around them: which candidates are legal, how many weight copies defeat the L2 cache, that an
invalid or only-marginally-faster config can never replace the default, and that the file it writes loads.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from engine.kernels.tuning import TunedConfigs

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tune_kernels.py"


@pytest.fixture(scope="module")
def tk():
    spec = importlib.util.spec_from_file_location("tune_kernels", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tune_kernels"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_gemm_candidates_are_all_legal_and_the_default_space_is_not_empty(tk):
    for k, n in ((2048, 2048), (2048, 256), (2048, 11008), (11008, 2048)):
        cands = tk.gemm_candidates(k, n, 128, tk.GEMM_SPACES["standard"])
        assert cands, (k, n)
        for c in cands:
            assert 128 % c["BLOCK_K"] == 0 and k % c["BLOCK_K"] == 0
            assert c["SPLIT_K"] <= k // c["BLOCK_K"] and c["BLOCK_N"] % 8 == 0
            assert set(c) == {"BLOCK_M", "BLOCK_N", "BLOCK_K", "SPLIT_K", "num_warps", "num_stages"}
    assert len(tk.gemm_candidates(2048, 11008, 128, tk.GEMM_SPACES["standard"])) == 3 * 2 * 5 * 2 * 3


def test_gemm_candidates_drop_illegal_tiles(tk):
    space = dict(BLOCK_N=(64,), BLOCK_K=(32, 96, 128), SPLIT_K=(1, 16), num_warps=(4,), num_stages=(3,))
    # K=512: BLOCK_K=96 divides neither K nor the group; BLOCK_K=32 has 16 K-blocks (16 splits legal);
    # BLOCK_K=128 has only 4 K-blocks, so 16 splits would leave programs with no work.
    got = tk.gemm_candidates(512, 256, 128, space)
    assert {(c["BLOCK_K"], c["SPLIT_K"]) for c in got} == {(32, 1), (32, 16), (128, 1)}


def test_attention_candidates_cover_the_product(tk):
    space = tk.ATTN_SPACES["standard"]
    cands = tk.attention_candidates(space)
    assert len(cands) == 3 * 3 * 3 * 4 and len({json.dumps(c, sort_keys=True) for c in cands}) == len(cands)


def test_working_set_is_several_times_larger_than_l2(tk):
    l2 = 24 << 20
    for per_call in (2 << 20, 12 << 20, 1 << 20):
        copies = tk.n_copies(per_call, l2)
        assert copies * per_call >= 4 * l2 or copies == 512
    assert tk.n_copies(12 << 20, l2) * (12 << 20) >= 128 << 20
    assert tk.n_copies(10**12, l2) == 2                           # never fewer than two copies
    assert tk.n_copies(1 << 20, l2, budget_bytes=10 << 20) == 10  # but never beyond the memory budget
    assert tk.n_copies(1 << 20, None) >= 128                      # unknown L2: assume a generous 32 MiB


def test_pick_best_keeps_the_default_unless_a_valid_config_is_clearly_faster(tk):
    default = {"cfg": {"d": 1}, "us": 100.0}
    cands = [{"cfg": {"a": 1}, "us": 98.0, "valid": True},       # 2% faster: inside the noise threshold
             {"cfg": {"b": 1}, "us": 10.0, "valid": False},      # much faster but WRONG: must never win
             {"cfg": {"c": 1}, "us": None, "valid": True}]       # failed to run
    r = tk.pick_best(cands, default, 0.03)
    assert r["picked"] == "default" and r["cfg"] == {"d": 1} and r["speedup"] == 1.0
    cands.append({"cfg": {"e": 1}, "us": 80.0, "valid": True})
    r = tk.pick_best(cands, default, 0.03)
    assert r["picked"] == "tuned" and r["cfg"] == {"e": 1} and r["speedup"] == pytest.approx(1.25)


def test_pick_best_needs_a_measured_default(tk):
    with pytest.raises(ValueError, match="default"):
        tk.pick_best([{"cfg": {}, "us": 1.0, "valid": True}], {"cfg": {}, "us": None}, 0.03)
    with pytest.raises(ValueError):
        tk.pick_best([], None, 0.03)


def test_without_a_gpu_the_script_refuses_to_pretend(tk, tmp_path, capsys):
    import torch

    if torch.cuda.is_available():
        pytest.skip("this machine has a GPU")
    assert tk.main(["--out-dir", str(tmp_path)]) == 2
    assert "no CUDA GPU" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())  # nothing written, no numbers invented


def test_smoke_run_writes_a_loadable_config_and_the_sweeps(tk, tmp_path):
    import torch

    if torch.cuda.is_available():
        pytest.skip("--smoke is the CPU path; on a GPU machine run the script for real")
    assert tk.main(["--smoke", "--out-dir", str(tmp_path)]) == 0
    tuned = TunedConfigs.load(tmp_path / "best_configs.json")
    assert set(tuned.gemm) == {(256, 256), (512, 128)}
    assert "NOT a measurement" in tuned.meta["device"]
    sweep = (tmp_path / "awq_gemm_sweep.csv").read_text().splitlines()
    assert sweep[0].startswith("K,N,") and len(sweep) > 5
    assert any("True" in line for line in sweep[1:])  # the default row is marked
    assert (tmp_path / "attention_sweep.csv").read_text().count("\n") >= 5
