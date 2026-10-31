#!/usr/bin/env bash
# Benchmark this engine against llama.cpp and vLLM on the same prompt token ids.
# Each backend runs in its own process (and, for the baselines, its own virtualenv)
# so that peak-VRAM readings are not contaminated by one another. A backend that
# cannot be installed or run is logged to results/benchmark_skips.log and skipped;
# nothing is ever filled in by hand.
#
# Environment overrides (defaults in brackets):
#   PYTHON [python3]                      interpreter with this package installed
#   MODEL  [Qwen/Qwen2.5-3B-Instruct-AWQ]
#   GGUF_REPO / GGUF_FILE                 llama.cpp weights (Q4_K_M by default)
#   LLAMACPP_PYTHON [.venvs/llamacpp/bin/python]   venv with llama-cpp-python (CUDA build)
#   VLLM_PYTHON     [.venvs/vllm/bin/python]       venv with vllm; its `vllm` binary is used
#   LOADER [autoawq]                      engine weight loader: autoawq (default) or safetensors
#                                         (explicit fallback if autoawq fails on your stack; it is
#                                         recorded in the CSV notes column, never chosen silently)
#   PROMPT_LEN [128] NEW_TOKENS [128] RUNS [10] WARMUP [2] MAX_SEQ [1024]
#   BACKENDS ["engine engine-reference-ops llamacpp vllm"]
#                                         engine variants: engine (eager), engine-graph (CUDA graph),
#                                         engine-tuned (tuned kernel configs), engine-graph-tuned (both)
#   TUNED_CONFIG [results/tuning/best_configs.json]   file from scripts/tune_kernels.py (the *-tuned backends)
#   RESULTS_DIR [results]                 where the CSV, skip log, logs and chart go (use a fresh directory for a
#                                         new run so committed results are not overwritten)
#   LLAMACPP_LD_LIBRARY_PATH              directories with libcudart.so.12 / libcublas*.so.12 for the prebuilt
#                                         llama-cpp-python CUDA wheel (it does not bundle them; the main venv's
#                                         torch install already has them under site-packages/nvidia/*/lib)
#   KEEP_CSV=1                            append to an existing benchmark.csv in RESULTS_DIR
set -u
cd "$(dirname "$0")/.." || exit 1

PYTHON=${PYTHON:-python3}
LOADER=${LOADER:-autoawq}
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct-AWQ}
GGUF_REPO=${GGUF_REPO:-Qwen/Qwen2.5-3B-Instruct-GGUF}
GGUF_FILE=${GGUF_FILE:-qwen2.5-3b-instruct-q4_k_m.gguf}
LLAMACPP_PYTHON=${LLAMACPP_PYTHON:-.venvs/llamacpp/bin/python}
VLLM_PYTHON=${VLLM_PYTHON:-.venvs/vllm/bin/python}
PROMPT_LEN=${PROMPT_LEN:-128}
NEW_TOKENS=${NEW_TOKENS:-128}
RUNS=${RUNS:-10}
WARMUP=${WARMUP:-2}
MAX_SEQ=${MAX_SEQ:-1024}
BACKENDS=${BACKENDS:-"engine engine-reference-ops llamacpp vllm"}
TUNED_CONFIG=${TUNED_CONFIG:-results/tuning/best_configs.json}
RESULTS_DIR=${RESULTS_DIR:-results}

OUT="$RESULTS_DIR/benchmark.csv"
SKIPS="$RESULTS_DIR/benchmark_skips.log"
LOGS="$RESULTS_DIR/logs"
mkdir -p results "$RESULTS_DIR" "$LOGS"
if [ "${KEEP_CSV:-0}" != "1" ]; then rm -f "$OUT" "$SKIPS"; fi
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

skip() { echo "$(date -u +%FT%TZ) SKIPPED $1: $2" | tee -a "$SKIPS"; }

# Record the environment the numbers came from.
{
  date -u +%FT%TZ
  nvidia-smi 2>&1 || echo "nvidia-smi not available"
  "$PYTHON" -m pip freeze 2>/dev/null | grep -i -E "^(torch|triton|transformers|autoawq|numpy|nvidia-ml-py)==" || true
  [ -x "$LLAMACPP_PYTHON" ] && "$LLAMACPP_PYTHON" -m pip freeze 2>/dev/null | grep -i "^llama" || true
  [ -x "$VLLM_PYTHON" ] && "$VLLM_PYTHON" -m pip freeze 2>/dev/null | grep -i -E "^(vllm|torch|triton)==" || true
} > "$LOGS/environment.txt" 2>&1

echo "== building the shared prompt ($PROMPT_LEN tokens)"
"$PYTHON" -m engine.bench prompt --model "$MODEL" --length "$PROMPT_LEN" --out results/prompt.json \
  || { echo "cannot build the prompt; aborting"; exit 1; }

COMMON=(--prompt-file results/prompt.json --new-tokens "$NEW_TOKENS" --runs "$RUNS" --warmup "$WARMUP"
        --max-seq-len "$MAX_SEQ" --out "$OUT")

for b in $BACKENDS; do
  echo "== backend: $b"
  case "$b" in
    engine|engine-reference-ops|engine-graph)
      "$PYTHON" -m engine.bench run --backend "$b" --model "$MODEL" --loader "$LOADER" "${COMMON[@]}" 2>&1 | tee "$LOGS/$b.log"
      [ "${PIPESTATUS[0]}" -eq 0 ] || skip "$b" "run failed, see $LOGS/$b.log"
      ;;
    engine-tuned|engine-graph-tuned)
      if [ ! -f "$TUNED_CONFIG" ]; then
        skip "$b" "no tuned config at $TUNED_CONFIG (run: python scripts/tune_kernels.py)"; continue
      fi
      "$PYTHON" -m engine.bench run --backend "$b" --model "$MODEL" --loader "$LOADER" --tuned-config "$TUNED_CONFIG" \
        "${COMMON[@]}" 2>&1 | tee "$LOGS/$b.log"
      [ "${PIPESTATUS[0]}" -eq 0 ] || skip "$b" "run failed, see $LOGS/$b.log"
      ;;
    llamacpp)
      if [ ! -x "$LLAMACPP_PYTHON" ]; then
        skip llamacpp "no interpreter at $LLAMACPP_PYTHON (run: make install-baselines)"; continue
      fi
      GGUF=$("$PYTHON" -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('$GGUF_REPO', '$GGUF_FILE'))" 2>"$LOGS/gguf_download.log") \
        || { skip llamacpp "could not download $GGUF_REPO/$GGUF_FILE, see $LOGS/gguf_download.log"; continue; }
      LD_LIBRARY_PATH="${LLAMACPP_LD_LIBRARY_PATH:-}${LLAMACPP_LD_LIBRARY_PATH:+:}${LD_LIBRARY_PATH:-}" \
        "$LLAMACPP_PYTHON" -m engine.bench run --backend llamacpp --gguf "$GGUF" "${COMMON[@]}" 2>&1 | tee "$LOGS/llamacpp.log"
      [ "${PIPESTATUS[0]}" -eq 0 ] || skip llamacpp "run failed, see $LOGS/llamacpp.log"
      ;;
    vllm)
      if [ ! -x "$VLLM_PYTHON" ]; then
        skip vllm "no interpreter at $VLLM_PYTHON; falling back to llama.cpp only (run: make install-baselines)"; continue
      fi
      "$VLLM_PYTHON" -m engine.bench run --backend vllm --model "$MODEL" \
        --vllm-bin "$(dirname "$VLLM_PYTHON")/vllm" --log "$LOGS/vllm_server.log" "${COMMON[@]}" 2>&1 | tee "$LOGS/vllm.log"
      [ "${PIPESTATUS[0]}" -eq 0 ] || skip vllm "run failed, see $LOGS/vllm.log and $LOGS/vllm_server.log"
      ;;
    *) skip "$b" "unknown backend" ;;
  esac
done

echo "== plotting"
"$PYTHON" scripts/plot_results.py --csv "$OUT" --out "$RESULTS_DIR/throughput.png" || skip plot "no rows to plot"
echo "== done. results: $OUT   skipped backends (if any): $SKIPS"
