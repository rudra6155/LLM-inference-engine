#!/usr/bin/env bash
# One-shot run of the OPTIMIZED engine on the RTX 4060: tests (including the real CUDA-graph tests), kernel
# tuning, and a benchmark of eager vs CUDA graph vs tuned vs both, against llama.cpp in the same session.
#
#   cd ~/LLM-inference-engine && git pull --ff-only origin main
#   source ~/venvs/engine/bin/activate
#   bash scripts/wsl_run_optimized.sh
#
#   * Everything is written to results/run_c/ (a fresh directory: committed results are never overwritten).
#   * No step aborts the run; every step's exit code is recorded in results/run_c/SUMMARY.txt.
#   * Nothing is filled in by hand: a step that fails leaves its numbers absent. If the CUDA graph cannot be
#     captured, the benchmark row says "cuda_graph=FALLBACK ..." in its notes and measures the eager loop.
#
# Environment overrides (all optional):
#   VENV [~/venvs/engine]   virtualenv to use if none is active
#   TUNE_SPACE [standard]   quick | standard | full   (size of the kernel sweep; every candidate is a separate Triton
#                           compile, so even "quick" takes minutes - run time is not estimated here)
#   SKIP_TUNE=1             reuse an existing $RES/tuning/best_configs.json (or results/tuning/best_configs.json)
#   SKIP_TESTS=1, SKIP_LLAMACPP=1
#   RUNS, NEW_TOKENS, PROMPT_LEN, STEPS   passed through to the benchmark / profile scripts
set -u -o pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
VENV=${VENV:-$HOME/venvs/engine}
RES=${RES:-$REPO/results/run_c}
TUNE_SPACE=${TUNE_SPACE:-standard}
STEPS=${STEPS:-10}
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 PIP_DISABLE_PIP_VERSION_CHECK=1
STATUS="$RES/status.tsv"
mkdir -p "$RES"
: > "$STATUS"

step() {  # step <name> <command...>   runs in this shell and records the exit code
  local name=$1; shift
  echo; echo "================ [$(date +%H:%M:%S)] $name ================"
  local t0=$SECONDS rc
  "$@"; rc=$?
  printf '%s\t%s\t%ss\n' "$name" "$rc" "$((SECONDS - t0))" >> "$STATUS"
  echo "---------------- $name -> exit $rc ($((SECONDS - t0))s)"
  return $rc
}

activate() {
  if [ -z "${VIRTUAL_ENV:-}" ]; then
    # shellcheck disable=SC1091
    [ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate" || { echo "ERROR: no virtualenv at $VENV (run scripts/wsl_run_all.sh first)"; return 1; }
  fi
  echo "python: $(command -v python) ($(python --version 2>&1))"
}

preflight() {
  activate || return 1
  git -C "$REPO" log -1 --format='repository HEAD: %H  %s'
  git -C "$REPO" status --short | head -5
  nvidia-smi || { echo "ERROR: the GPU is not visible in WSL2"; return 1; }
  python - <<'PY' || return 1
import torch, triton
print("torch", torch.__version__, "| triton", triton.__version__, "| CUDA", torch.version.cuda)
ok = torch.cuda.is_available()
print("cuda available:", ok)
if ok:
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name}: {p.multi_processor_count} SMs, {p.total_memory / 2**30:.2f} GiB, L2 = {getattr(p, 'L2_cache_size', 'unknown')} B")
raise SystemExit(0 if ok else 1)
PY
}

install_engine() { ( cd "$REPO" && make install ) 2>&1 | tail -15; return "${PIPESTATUS[0]}"; }

run_tests() {  # includes the CUDA-graph tests that are skipped on machines without a GPU
  ( cd "$REPO" && python -m pytest tests -k "not real_checkpoint" -q -rs --tb=short ) 2>&1 | tee "$RES/pytest.log"
  return "${PIPESTATUS[0]}"
}

roofline() { ( cd "$REPO" && python scripts/roofline.py --results "$RES" ) 2>&1 | tee "$RES/roofline.log"; return "${PIPESTATUS[0]}"; }

tune() {
  ( cd "$REPO" && python scripts/tune_kernels.py --space "$TUNE_SPACE" --out-dir "$RES/tuning" ) 2>&1 | tee "$RES/tune.log"
  return "${PIPESTATUS[0]}"
}

cuda_lib_dirs() {  # dirs holding libcudart.so.12 / libcublas*.so.12 that torch's pip wheels installed
  python - <<'PY'
import glob, os, site
dirs = []
for sp in site.getsitepackages():
    for sub in ("cuda_runtime", "cublas"):
        dirs += glob.glob(os.path.join(sp, "nvidia", sub, "lib"))
print(os.pathsep.join(dirs))
PY
}

benchmark() {
  local tuned="$RES/tuning/best_configs.json" backends="engine engine-graph"
  [ -f "$tuned" ] || tuned="$REPO/results/tuning/best_configs.json"
  if [ -f "$tuned" ]; then backends="$backends engine-tuned engine-graph-tuned"
  else echo "NOTE: no tuned config found; the *-tuned variants are skipped"; fi
  if [ "${SKIP_LLAMACPP:-0}" != 1 ] && [ -x "$REPO/.venvs/llamacpp/bin/python" ]; then backends="$backends llamacpp"
  else echo "NOTE: llama.cpp not benchmarked (SKIP_LLAMACPP=1 or no $REPO/.venvs/llamacpp)"; fi
  echo "backends: $backends   tuned config: $tuned"
  ( cd "$REPO" && RESULTS_DIR="$RES" TUNED_CONFIG="$tuned" BACKENDS="$backends" \
      LLAMACPP_LD_LIBRARY_PATH="$(cuda_lib_dirs)" bash scripts/benchmark.sh ) 2>&1 | tee "$RES/benchmark.log"
  return "${PIPESTATUS[0]}"
}

profile() {  # CUPTI-free eager-vs-graph timing of one decode step, untuned and tuned
  local tuned="$RES/tuning/best_configs.json" rc=0
  [ -f "$tuned" ] || tuned="$REPO/results/tuning/best_configs.json"
  ( cd "$REPO" && mkdir -p "$RES/profile_untuned" && cp "$RES/hw_ceilings.json" "$RES/profile_untuned/" 2>/dev/null
    python scripts/profile_decode.py --variant custom --steps "$STEPS" --prompt-file "$REPO/results/prompt.json" \
      --results "$RES/profile_untuned" ) 2>&1 | tee "$RES/profile_untuned.log"; rc=${PIPESTATUS[0]}
  if [ -f "$tuned" ]; then
    ( cd "$REPO" && mkdir -p "$RES/profile_tuned" && cp "$RES/hw_ceilings.json" "$RES/profile_tuned/" 2>/dev/null
      python scripts/profile_decode.py --variant custom --steps "$STEPS" --prompt-file "$REPO/results/prompt.json" \
        --results "$RES/profile_tuned" --tuned-config "$tuned" ) 2>&1 | tee "$RES/profile_tuned.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
  fi
  return $rc
}

summarize() {
  {
    echo "################ SUMMARY  $(date -u +%FT%TZ) ################"
    echo; echo "== steps: name <TAB> exit code <TAB> seconds =="; cat "$STATUS"
    echo; echo "== git =="; git -C "$REPO" log -1 --format='%H  %s'; git -C "$REPO" status --short | head -20
    echo; echo "== tests =="; grep -E "passed|failed|error|skipped" "$RES/pytest.log" 2>/dev/null | tail -8
    echo; echo "== benchmark.csv =="
    python - "$RES/benchmark.csv" <<'PY' 2>/dev/null || echo "(absent)"
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print(f"{'backend':<22}{'tok/s':>8}{'ttft ms':>9}{'itl p50':>9}{'itl p99':>9}{'itl max':>9}{'VRAM MiB':>10}  notes")
for r in rows:
    print(f"{r['backend']:<22}{float(r['decode_tok_s']):>8.2f}{float(r['ttft_ms_p50']):>9.1f}{float(r['itl_ms_p50']):>9.1f}"
          f"{float(r['itl_ms_p99']):>9.1f}{float(r['itl_ms_max']):>9.1f}{float(r['peak_vram_mib']):>10.0f}  {r['notes']}")
PY
    echo; echo "== benchmark_skips.log =="; cat "$RES/benchmark_skips.log" 2>/dev/null || echo "(none)"
    echo; echo "== hw_ceilings.json =="; cat "$RES/hw_ceilings.json" 2>/dev/null || echo "(absent)"
    echo; echo "== tuning: default vs chosen (see $RES/tuning/best_configs.json) =="
    sed -n '/== summary/,$p' "$RES/tune.log" 2>/dev/null || echo "(absent)"
    for v in untuned tuned; do
      echo; echo "== profile_$v (CUPTI-free) =="
      python - "$RES/profile_$v/profile_custom_summary.json" <<'PY' 2>/dev/null || echo "(absent)"
import json, sys
d = json.load(open(sys.argv[1]))
for k in ("eager_ms_per_step", "graph_replay_ms_per_step", "launch_overhead_fraction_of_eager",
          "useful_bandwidth_GBs_graph", "useful_bandwidth_pct_of_measured_copy_bw_graph", "timing_errors"):
    print(f"  {k}: {d.get(k)}")
PY
    done
    echo; echo "== environment =="; python -m pip freeze 2>/dev/null | grep -i -E "^(torch|triton|transformers|autoawq|numpy)==" || true
  } > "$RES/SUMMARY.txt" 2>&1
  cat "$RES/SUMMARY.txt"
}

step preflight preflight || { echo "preflight failed; nothing else can run"; summarize; exit 1; }
step install install_engine
[ "${SKIP_TESTS:-0}" = 1 ] || step tests run_tests
step roofline roofline
if [ "${SKIP_TUNE:-0}" = 1 ]; then echo "SKIP_TUNE=1: not tuning"; else step tune tune; fi
step benchmark benchmark
step profile profile
summarize
echo
echo "Done. Results are in $RES (SUMMARY.txt first). To publish them:"
echo "  cd $REPO && git checkout -b gpu-results-run-c && git add results/run_c && git commit -m 'Add Run C results' && git push -u origin gpu-results-run-c"
