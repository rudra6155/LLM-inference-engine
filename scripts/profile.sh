#!/usr/bin/env bash
# Profile the decode phase: custom kernels vs the PyTorch-ops baseline.
#
#  1. torch.profiler (needs only the GPU): Chrome traces in results/traces/ and per-kernel
#     GPU time / derived "useful bandwidth" in results/profile_*.csv|json.
#  2. Nsight Compute (needs `ncu` and permission to read GPU performance counters):
#     DRAM throughput % and DRAM bytes per kernel -> results/logs/ncu_<variant>.csv.
#     If `ncu` is missing or refuses, that is logged and the report is built without
#     counters - nothing is substituted.
#  3. scripts/profile_report.py -> results/profile_summary.csv, results/dram_throughput.png
#
# Overrides: PYTHON [python3], MODEL, LOADER [autoawq | safetensors], STEPS [20], NCU_STEPS [2]
set -u
cd "$(dirname "$0")/.." || exit 1
PYTHON=${PYTHON:-python3}
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct-AWQ}
LOADER=${LOADER:-autoawq}
STEPS=${STEPS:-20}
NCU_STEPS=${NCU_STEPS:-2}
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p results/logs results/traces
SKIPS=results/profile_skips.log
: > "$SKIPS"

# The roofline run measures copy bandwidth, which the profile summary uses as its reference.
"$PYTHON" scripts/roofline.py || echo "roofline measurement failed" | tee -a "$SKIPS"
[ -f results/prompt.json ] || "$PYTHON" -m engine.bench prompt --model "$MODEL" --length 128 --out results/prompt.json

for v in custom reference; do
  echo "== torch.profiler: $v"
  "$PYTHON" scripts/profile_decode.py --variant "$v" --model "$MODEL" --loader "$LOADER" --steps "$STEPS" \
    || echo "torch.profiler run for $v failed" | tee -a "$SKIPS"
done

if command -v ncu >/dev/null 2>&1; then
  METRICS=$("$PYTHON" scripts/profile_report.py --print-ncu-metrics)
  for v in custom reference; do
    echo "== Nsight Compute: $v ($METRICS)"
    ncu --profile-from-start off --csv --page raw --print-units base --metrics "$METRICS" \
      "$PYTHON" scripts/profile_decode.py --variant "$v" --model "$MODEL" --loader "$LOADER" --ncu --steps "$NCU_STEPS" \
      > "results/logs/ncu_$v.csv" 2> "results/logs/ncu_$v.err" \
      || echo "ncu run for $v failed (see results/logs/ncu_$v.err; counters may need elevated permissions)" | tee -a "$SKIPS"
  done
else
  echo "ncu not found: DRAM-throughput counters were not collected" | tee -a "$SKIPS"
fi

"$PYTHON" scripts/profile_report.py --ncu-steps "$NCU_STEPS" || echo "report failed" | tee -a "$SKIPS"
echo "== done. See results/profile_summary.csv, results/dram_throughput.png, results/traces/, $SKIPS"
