#!/usr/bin/env bash
# One-shot run for WSL2: install, test, measure, benchmark, profile, and bundle every result.
#
#   git clone -b main https://github.com/rudra6155/LLM-inference-engine.git ~/LLM-inference-engine
#   bash ~/LLM-inference-engine/scripts/wsl_run_all.sh
#
#   * No step aborts the run. Every step's exit code is recorded in SUMMARY.txt.
#   * Safe to re-run (reuses the clone, the virtualenvs and downloaded models).
#   * Nothing is ever filled in by hand: a step that fails leaves its numbers absent.
#
# Environment overrides (all optional):
#   REPO, VENV            where the clone / main virtualenv live
#   SKIP_APT=1            do not touch apt
#   SKIP_BASELINES=1      engine only (no llama.cpp / vLLM)
#   RUNS, NEW_TOKENS, PROMPT_LEN, STEPS   passed through to the benchmark / profile scripts
set -u -o pipefail

REPO_URL=https://github.com/rudra6155/LLM-inference-engine.git
BRANCH=main
REVIEWED_COMMIT=3002418ad98f95d78781f7f42b34e550fbf2c804   # the code commit that was reviewed
# Default: the repository this script lives in (so running it from the clone just works).
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd || echo "$HOME/LLM-inference-engine")}
VENV=${VENV:-$HOME/venvs/engine}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${OUT:-$HOME/engine_run_$STAMP}
LOADER=${LOADER:-autoawq}
export LOADER
export PYTHONUNBUFFERED=1 PIP_PROGRESS_BAR=off HF_HUB_DISABLE_PROGRESS_BARS=1 PIP_DISABLE_PIP_VERSION_CHECK=1

GPU_OK=0
REAL_CKPT_OK=0
LLAMA_OK=0
VLLM_OK=0

# --------------------------------------------------------------------------- helpers
STATUS="$OUT/status.tsv"

step() {  # step <name> <function or command...>  -- runs in this shell, records the exit code
  local name=$1; shift
  echo; echo "================ [$(date +%H:%M:%S)] $name ================"
  local t0=$SECONDS rc
  "$@"; rc=$?
  printf '%s\t%s\t%ss\n' "$name" "$rc" "$((SECONDS - t0))" >> "$STATUS"
  echo "---------------- $name -> exit $rc ($((SECONDS - t0))s)"
  return $rc
}

note_skip() {  # note_skip <what> <why>  (also lands in the benchmark skip log the repo reads)
  echo "SKIPPED $1: $2"
  mkdir -p "$REPO/results"
  echo "$(date -u +%FT%TZ) SKIPPED $1: $2" >> "$REPO/results/benchmark_skips.log"
}

free_gib() { df -Pk "$1" | awk 'NR==2 {printf "%d", $4/1024/1024}'; }

# --------------------------------------------------------------------------- steps
preflight() {
  echo "date: $(date -u +%FT%TZ)"; uname -a; head -3 /etc/os-release 2>/dev/null
  if grep -qi microsoft /proc/version 2>/dev/null; then echo "WSL: yes"; else echo "WSL: NO - this script targets WSL2; continuing anyway"; fi
  echo "python3: $(python3 --version 2>&1)"
  if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "ERROR: Python >= 3.10 is required"; return 1
  fi
  echo "cpu threads: $(nproc)"
  echo "memory (MiB): total $(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo), available $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)"
  if [ "$(awk '/MemAvailable/ {print $2}' /proc/meminfo)" -ge 10485760 ]; then REAL_CKPT_OK=1; fi
  echo "disk free in \$HOME: $(free_gib "$HOME") GiB"
  echo "gcc: $(gcc --version 2>&1 | head -1)"; echo "nvcc: $(nvcc --version 2>&1 | tail -1)"
  echo "--- nvidia-smi ---"
  if nvidia-smi; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.mem --format=csv 2>&1 || true
  else
    echo "nvidia-smi failed: the GPU is not visible inside WSL2 (see the notes printed at the end)"
  fi
  [ "$(free_gib "$HOME")" -ge 15 ] || { echo "ERROR: less than 15 GiB free in \$HOME"; return 1; }
  return 0
}

apt_prereqs() {
  [ "${SKIP_APT:-0}" = 1 ] && { echo "SKIP_APT=1: not touching apt"; return 0; }
  local need=() p SUDO=""
  for p in build-essential python3-dev python3-venv git curl; do dpkg -s "$p" >/dev/null 2>&1 || need+=("$p"); done
  if [ ${#need[@]} -eq 0 ]; then echo "apt prerequisites already present"; return 0; fi
  [ "$(id -u)" -ne 0 ] && SUDO=sudo
  echo "installing: ${need[*]}   (your sudo password may be requested)"
  $SUDO apt-get update && $SUDO apt-get install -y "${need[@]}"
}

get_repo() {
  if [ -d "$REPO/.git" ]; then
    # Only fetch: merging here could rewrite this very script while bash is reading it.
    git -C "$REPO" fetch origin "$BRANCH" || echo "WARNING: could not fetch (offline?); using the local checkout"
    local behind; behind=$(git -C "$REPO" rev-list --count "HEAD..origin/$BRANCH" 2>/dev/null || echo 0)
    [ "$behind" = 0 ] || echo "NOTE: the local checkout is $behind commit(s) behind origin/$BRANCH. Run 'git -C $REPO pull --ff-only' and re-run to use the latest."
  else
    git clone -b "$BRANCH" "$REPO_URL" "$REPO" || return 1
  fi
  local head; head=$(git -C "$REPO" rev-parse HEAD)
  echo "HEAD: $head"
  if git -C "$REPO" merge-base --is-ancestor "$REVIEWED_COMMIT" HEAD 2>/dev/null; then
    echo "includes the reviewed commit $REVIEWED_COMMIT"
  else
    echo "NOTE: HEAD does not contain the reviewed commit $REVIEWED_COMMIT"
  fi
}

setup_env() {
  mkdir -p "$(dirname "$VENV")"
  [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV" || return 1
  # shellcheck disable=SC1091
  source "$VENV/bin/activate" || return 1
  python -m pip install --upgrade pip || return 1
  ( cd "$REPO" && make install )
}

gpu_check() {
  command -v nvidia-smi >/dev/null && nvidia-smi -L || return 1
  ( cd "$REPO" && python - <<'PY'
import sys, torch
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
ok = torch.cuda.is_available()
print("torch.cuda.is_available():", ok)
if not ok:
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
print(f"device: {p.name} | compute capability {p.major}.{p.minor} | {p.multi_processor_count} SMs | {p.total_memory / 2**30:.2f} GiB")
x = torch.randn(512, 512, device="cuda"); torch.cuda.synchronize()
print("a CUDA matmul runs:", bool(torch.isfinite((x @ x).sum()).item()))
PY
  ) && GPU_OK=1
}

probe_autoawq() {
  ( cd "$REPO" && python - <<'PY'
import time, torch
from engine.model import load_awq_model
t = time.time()
m = load_awq_model("Qwen/Qwen2.5-3B-Instruct-AWQ", loader="autoawq", device="cuda", max_seq_len=64)
torch.cuda.synchronize()
print(f"AUTOAWQ_LOAD_OK in {time.time() - t:.0f}s; embeddings on {m.embed.device}; "
      f"GPU memory allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
PY
  ) 2>&1 | tee "$OUT/autoawq_probe.log"
  return "${PIPESTATUS[0]}"
}

run_tests() {
  cd "$REPO" || return 1
  local rc
  export PYTEST_ADDOPTS="-rA --tb=short"
  if [ "$GPU_OK" = 1 ] && [ "$REAL_CKPT_OK" != 1 ]; then
    echo "NOTE: less than 10 GiB of RAM is available to WSL2 -> the real-checkpoint test (needs ~7.5 GiB for its"
    echo "      HuggingFace reference) is DESELECTED to avoid an out-of-memory kill. Everything else runs."
    python -m pytest tests -k "not real_checkpoint" 2>&1 | tee "$OUT/make_test.log"; rc=${PIPESTATUS[0]}
  else
    make test 2>&1 | tee "$OUT/make_test.log"; rc=${PIPESTATUS[0]}
  fi
  unset PYTEST_ADDOPTS
  return "$rc"
}

run_cmd_logged() {  # run_cmd_logged <logname> <command...>
  local log=$1; shift
  ( cd "$REPO" && "$@" ) 2>&1 | tee "$OUT/$log.log"
  return "${PIPESTATUS[0]}"
}

bench_engine() {
  ( cd "$REPO" && BACKENDS="engine engine-reference-ops" bash scripts/benchmark.sh ) 2>&1 | tee "$OUT/benchmark_engine.log"
  return "${PIPESTATUS[0]}"
}

profile_all() {
  ( cd "$REPO" && STEPS="${STEPS:-10}" NCU_STEPS="${NCU_STEPS:-1}" bash scripts/profile.sh ) 2>&1 | tee "$OUT/profile.log"
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

llama_smoke() {
  echo "LD_LIBRARY_PATH for llama.cpp: ${LLAMACPP_LD_LIBRARY_PATH:-<empty>}"
  LD_LIBRARY_PATH="${LLAMACPP_LD_LIBRARY_PATH:-}${LLAMACPP_LD_LIBRARY_PATH:+:}${LD_LIBRARY_PATH:-}" "$1/bin/python" - <<'PY'
import importlib.metadata as md
import llama_cpp
print("llama-cpp-python", md.version("llama_cpp_python"))
ok = llama_cpp.llama_supports_gpu_offload()
print("GPU offload supported:", ok)
raise SystemExit(0 if ok else 1)
PY
}

install_llamacpp() {
  local V="$REPO/.venvs/llamacpp"
  LLAMACPP_LD_LIBRARY_PATH=$(cuda_lib_dirs); export LLAMACPP_LD_LIBRARY_PATH
  [ "$(free_gib "$HOME")" -ge 12 ] || { note_skip llamacpp "less than 12 GiB free for the CUDA wheel"; return 2; }
  [ -x "$V/bin/python" ] || python3 -m venv "$V" || return 1
  "$V/bin/python" -m pip install --upgrade pip
  echo "--- try 1: prebuilt CUDA 12.4 wheel (no nvcc needed) ---"
  if "$V/bin/python" -m pip install --only-binary=llama-cpp-python "llama-cpp-python==0.3.36" numpy nvidia-ml-py \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 && llama_smoke "$V"; then
    LLAMA_OK=1; echo "llama.cpp: prebuilt wheel works"; return 0
  fi
  echo "--- try 2: build from source with CUDA (needs nvcc) ---"
  if command -v nvcc >/dev/null 2>&1; then
    if CMAKE_ARGS="-DGGML_CUDA=on" "$V/bin/python" -m pip install --no-binary=llama-cpp-python --force-reinstall \
          "llama-cpp-python==0.3.36" numpy nvidia-ml-py && llama_smoke "$V"; then
      LLAMA_OK=1; echo "llama.cpp: source build works"; return 0
    fi
  else
    echo "nvcc not found, cannot build from source"
  fi
  note_skip llamacpp "no working CUDA build of llama-cpp-python could be installed (see run.log)"
  return 1
}

install_vllm() {
  local V="$REPO/.venvs/vllm" cuda_ver cuda_major
  if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
    note_skip vllm "vLLM 0.32.0 requires Python >= 3.11 (found $(python3 --version 2>&1)); falling back to llama.cpp only"; return 2
  fi
  cuda_ver=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
  cuda_major=${cuda_ver%%.*}
  if [ -z "$cuda_major" ] || [ "$cuda_major" -lt 13 ]; then
    note_skip vllm "vLLM 0.32.0 pins torch 2.13.0, built for CUDA 13.0; the installed driver supports CUDA ${cuda_ver:-unknown}. Falling back to llama.cpp only"
    return 2
  fi
  [ "$(free_gib "$HOME")" -ge 20 ] || { note_skip vllm "less than 20 GiB free for vLLM + torch"; return 2; }
  [ -x "$V/bin/python" ] || python3 -m venv "$V" || return 1
  "$V/bin/python" -m pip install --upgrade pip
  if "$V/bin/python" -m pip install "vllm==0.32.0" nvidia-ml-py && "$V/bin/python" - <<'PY'
import torch, vllm
print("vllm", vllm.__version__, "| torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then VLLM_OK=1; return 0; fi
  note_skip vllm "installation or CUDA smoke test failed (see run.log)"
  return 1
}

bench_baselines() {
  local b=""
  [ "$LLAMA_OK" = 1 ] && b="llamacpp"
  [ "$VLLM_OK" = 1 ] && b="$b vllm"
  [ -n "$b" ] || { echo "no baseline is installed; nothing to benchmark"; return 0; }
  ( cd "$REPO" && KEEP_CSV=1 BACKENDS="$b" bash scripts/benchmark.sh ) 2>&1 | tee "$OUT/benchmark_baselines.log"
  return "${PIPESTATUS[0]}"
}

package() {
  cd "$REPO" || return 1
  cp "$OUT/run.log" "$OUT/run_copy.log" 2>/dev/null || true
  {
    echo "################ SUMMARY  $(date -u +%FT%TZ) ################"
    echo; echo "== steps: name <TAB> exit code <TAB> seconds =="; cat "$STATUS"
    echo; echo "== flags =="; echo "GPU_OK=$GPU_OK REAL_CKPT_OK=$REAL_CKPT_OK LLAMA_OK=$LLAMA_OK VLLM_OK=$VLLM_OK LOADER=$LOADER ENGINE_LOADER=${ENGINE_LOADER:-}"
    [ -f "$OUT/LOADER_FALLBACK.txt" ] && { echo; cat "$OUT/LOADER_FALLBACK.txt"; }
    echo; echo "== git =="; git rev-parse HEAD; git status --short | head -30
    echo; echo "== environment =="; head -3 /etc/os-release 2>/dev/null; uname -r; python --version
    python -m pip freeze 2>/dev/null | grep -i -E '^(torch|triton|transformers|autoawq|numpy|accelerate|datasets|huggingface|tokenizers|safetensors|nvidia-ml-py|pytest|ninja)=='
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.mem --format=csv 2>&1 | head -3
    echo; echo "== autoawq probe =="; tail -5 "$OUT/autoawq_probe.log" 2>/dev/null || echo "(not run)"
    echo; echo "== test result lines (make test) =="
    grep -E "^(FAILED|ERROR)|passed|failed|engine loader=|^SKIPPED" "$OUT/make_test.log" 2>/dev/null | tail -70
    [ -f "$OUT/test_real_checkpoint_fallback.log" ] && { echo; echo "== real-checkpoint test re-run with the safetensors loader =="; grep -E "engine loader=|passed|failed|FAILED|Error" "$OUT/test_real_checkpoint_fallback.log" | tail -10; }
    for f in results/benchmark.csv results/benchmark_skips.log results/hw_ceilings.json results/roofline_measured.json \
             results/kernel_max_diff.csv results/cpp_op_bench.csv results/profile_custom_summary.json \
             results/profile_reference_summary.json results/profile_summary.csv results/profile_skips.log; do
      echo; echo "== $f =="
      if [ ! -f "$f" ]; then echo "(absent)"
      elif [ "$f" -nt "$OUT/.start" ]; then cat "$f"
      else echo "(not produced by this run: this is the version committed in the repo, so it is NOT shown)"; fi
    done
    for f in results/logs/engine.log results/logs/engine-reference-ops.log results/logs/llamacpp.log results/logs/vllm.log results/logs/ncu_custom.err; do
      [ -f "$f" ] && [ "$f" -nt "$OUT/.start" ] && { echo; echo "== tail of $f =="; tail -15 "$f"; }
    done
    echo; echo "== trace files =="; ls -la results/traces 2>/dev/null || echo "(none)"
  } > "$OUT/SUMMARY.txt" 2>&1
  local extra=(SUMMARY.txt status.tsv run_copy.log) f b
  for f in "$OUT"/*.log; do
    [ -e "$f" ] || continue
    b=$(basename "$f"); case "$b" in run.log|run_copy.log) ;; *) extra+=("$b") ;; esac
  done
  tar czf "$OUT/engine_results.tar.gz" --exclude='results/traces' -C "$REPO" results -C "$OUT" "${extra[@]}" 2>&1 | tail -3
  [ -d "$REPO/results/traces" ] && tar czf "$OUT/traces.tar.gz" -C "$REPO" results/traces
  return 0
}

final_notes() {
  echo
  echo "####################################################################"
  echo "DONE.   Step results:"; column -t -s "$(printf '\t')" "$STATUS" 2>/dev/null || cat "$STATUS"
  echo
  echo "Send me this one file (paste its contents):   $OUT/SUMMARY.txt"
  echo "Full bundle (logs + results, no traces):      $OUT/engine_results.tar.gz   ($(du -h "$OUT/engine_results.tar.gz" 2>/dev/null | cut -f1))"
  [ -f "$OUT/traces.tar.gz" ] && echo "Profiler traces (large, only if I ask):        $OUT/traces.tar.gz   ($(du -h "$OUT/traces.tar.gz" | cut -f1))"
  echo
  if [ "$GPU_OK" != 1 ]; then
    echo "!! The GPU was not usable from this environment, so the GPU steps were skipped."
    echo "   Check, in this order:  (1) 'nvidia-smi' works in WSL2   (2) the Windows NVIDIA driver is current"
    echo "   (3) the line 'torch.cuda.is_available()' in $OUT/run.log.  Then re-run this script."
  fi
  echo "####################################################################"
}

# --------------------------------------------------------------------------- main
main() {
  mkdir -p "$OUT"; : > "$STATUS"; touch "$OUT/.start"; sleep 1
  exec > >(tee -a "$OUT/run.log") 2>&1
  echo "engine one-shot run  $STAMP   (output directory: $OUT)"
  echo "Reminders: laptop plugged in and in a high-performance power mode; no other GPU-heavy apps running."

  step preflight preflight || { final_notes; return 1; }
  step apt_prereqs apt_prereqs
  step get_repo get_repo || { final_notes; return 1; }
  step setup_env setup_env || { echo "install failed: see run.log"; step package package; final_notes; return 1; }
  step gpu_check gpu_check

  if [ "$GPU_OK" = 1 ]; then
    if ! step autoawq_probe probe_autoawq; then
      export LOADER=safetensors ENGINE_LOADER=safetensors
      {
        echo "== LOADER FALLBACK =="
        echo "autoawq could not load the checkpoint on this stack (see autoawq_probe.log)."
        echo "Everything after this point used the explicit safetensors loader (LOADER=safetensors)."
        echo "The 'loaded with autoawq' claim therefore remains UNVERIFIED."
      } | tee "$OUT/LOADER_FALLBACK.txt"
    fi
  fi

  step make_test run_tests
  if [ "$GPU_OK" = 1 ] && [ "$REAL_CKPT_OK" = 1 ] && grep -q "FAILED tests/test_engine.py::test_real_checkpoint" "$OUT/make_test.log" 2>/dev/null \
     && [ "${ENGINE_LOADER:-}" != safetensors ]; then
    step real_checkpoint_fallback run_cmd_logged test_real_checkpoint_fallback env ENGINE_LOADER=safetensors python -m pytest tests/test_engine.py -k real_checkpoint -s -q
  fi

  if [ "$GPU_OK" = 1 ]; then
    step kernel_max_diff run_cmd_logged kernel_max_diff python scripts/kernel_max_diff.py
    step cpp_op_bench run_cmd_logged cpp_op_bench python scripts/bench_cpp_op.py
    step roofline run_cmd_logged roofline python scripts/roofline.py
    step bench_engine bench_engine
    step profile profile_all
    if [ "${SKIP_BASELINES:-0}" != 1 ]; then
      step install_llamacpp install_llamacpp
      step install_vllm install_vllm
      step bench_baselines bench_baselines
    else
      echo "SKIP_BASELINES=1: baselines not run"
    fi
  fi

  step package package
  final_notes
  sleep 1
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
