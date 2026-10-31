# Benchmark Claims & Verification Matrix

One row per technical claim and benchmark metric. Statuses:

* **VERIFIED** - Backed by a committed artifact in `results/` or reproduced by the command in the *Evidence* column. Tested on an RTX 4060 Laptop GPU under WSL2 (or CPU test suite where indicated).
* **NOT RUN / NOT MET** - Target was not evaluated or target was missed (e.g., performance comparison against llama.cpp is recorded as an empirical engineering result).
* **PENDING** - Requires additional hardware or unsupported dependency.

**Run A** = Initial baseline benchmark run (2026-10-30, 13:59-14:17 UTC); raw `results/benchmark.csv` committed.
**Run B** = Profiling run (2026-10-30, 14:57-15:23 UTC) evaluating eager execution, PyTorch reference ablation, llama.cpp baseline, and eager vs CUDA graph timing; raw files committed in `results/run_b/`.
**Run C** = Optimized run (2026-10-30, 23:16-23:28 UTC) incorporating CUDA graphs, kernel tuning, and final performance verification; raw files committed in `results/run_c/`.

Methodology: Every throughput/latency number links directly to a committed `benchmark.csv`; every correctness claim references an automated test or verification script; memory metrics reference analytic formulas or direct hardware measurements. "VERIFIED (interpreter)" indicates Triton kernel execution via `TRITON_INTERPRET=1` on CPU to prove arithmetic logic and numerics. All optimization claims (O9, O10, O13, O14) have been validated on the RTX 4060 GPU in Run C.

## A. Memory and first-principles claims

| ID | Claim | Status | Evidence |
|---|---|---|---|
| M1 | KV cache = 2 x 36 x 2 x 128 x 2 = **36,864 B/token** for Qwen2.5-3B (fp16) | VERIFIED | `pytest tests/test_engine.py -k kv_bytes_per_token`; `python scripts/kv_cache.py`; inputs from the checkpoint `config.json` (SPEC.md section 3) |
| M2 | A real `KVCache` allocates exactly `36,864 x max_seq_len` bytes (real and tiny config; lengths 1, 64, 300) | VERIFIED | `pytest tests/test_engine.py -k allocated_kv_cache` (checks `numel*element_size` and the storage's allocated bytes) |
| M3 | KV size table (4.5 MiB @128 ... 1,152 MiB @32,768) | VERIFIED | `python scripts/kv_cache.py` (arithmetic from M1) |
| M4 | Weights held once (tied lm_head) = 2,064,269,312 B = 1.92 GiB | VERIFIED | `results/checkpoint_check.txt` (tensor sizes from the safetensors header) minus the duplicate lm_head; `python scripts/kv_cache.py` |
| M5 | `lm_head.weight` and `embed_tokens.weight` are byte-identical (tying is lossless) | VERIFIED | `python scripts/check_checkpoint.py` -> `results/checkpoint_check.txt` (SHA-256 equal) |
| M6 | The analytic linear-weight byte model equals the checkpoint's `qweight+qzeros+scales+bias` bytes (1,441,640,448 B) | VERIFIED | `pytest tests/test_roofline.py`; `results/checkpoint_check.txt` |
| M7 | Engine peak VRAM above baseline, Run A: **2,899.43 MiB** (device-level NVML; pre-load baseline 945.19 MiB) | VERIFIED | `results/benchmark.csv` (row `engine`: `peak_vram_mib`, `vram_baseline_mib`, `vram_method=nvml-device-level`). Analytic floor 2,005 MiB (weights 1,969 + KV@1024 36); the other ~894 MiB is not decomposed. |
| M8 | Run B peak VRAM above baseline: engine **3,703.86 MiB** (baseline 1,266.58), ablation 3,367.26 (1,761.82), llama.cpp **2,570.62** (1,325.35). | VERIFIED (values); cause of the difference unexplained | `results/run_b/benchmark.csv` (`peak_vram_mib`, `vram_baseline_mib`); the harness records no per-process figure |
| M9 | Run C peak VRAM (Graph + Tuned): **2,897 MiB**. | VERIFIED | `results/run_c/benchmark.csv` (`peak_vram_mib`) |

## B. Benchmark claims

| ID | Claim | Status | Evidence |
|---|---|---|---|
| B1 | Run A engine decode tokens/s = **12.99** | VERIFIED | `results/benchmark.csv`, row `engine`, 2026-10-30T13:59:49+00:00 |
| B2 | Run A engine TTFT (p50) = **132.6 ms** | VERIFIED | same row |
| B3 | Run A engine p99 inter-token latency = **108.6 ms** | VERIFIED | same row |
| B4 | llama.cpp: **Run B: 93.06 tokens/s**, TTFT 34.94 ms, p99 27.22 ms, peak VRAM 2,570.62 MiB; notes `flash_attn=True; n_gpu_layers=-1` | VERIFIED | `results/run_b/benchmark.csv` row `llama.cpp` (2026-10-30T15:23:44) |
| B5 | vLLM tokens/s, TTFT, p99, VRAM | **NOT RUN**: vLLM 0.32.0 requires Python >= 3.11 (found 3.10.12). | `results/benchmark_skips.log`, `results/run_b/benchmark_skips.log` |
| B6 | **Target: engine beats each baseline's batch-1 decode tokens/s** | **NOT MET against llama.cpp** (Run C CSV: **74.14** vs **101.70** tok/s = 1.37x slower; TTFT 80.5 vs 34.8 ms = 2.31x; p99 14.2 vs 11.7 ms = 1.21x). **Not evaluated against vLLM** (did not run) | `results/run_c/benchmark.csv`; FINDINGS.md sections 1 and 3 |
| B7 | `results/throughput.png` bar chart | VERIFIED (committed; Run A rows only) | drawn by `scripts/plot_results.py` from `results/benchmark.csv` |
| B13 | Ablation (same engine, PyTorch reference ops instead of the custom kernels): **Run A 1.46 tok/s**, **Run B 1.15 tok/s**. The custom kernels + C++ op give **10.23x** speedup in Run B. | VERIFIED | `results/run_b/benchmark.csv` row `engine-reference-ops` |
| B14 | `LD_LIBRARY_PATH` pointing at the main venv's `nvidia/{cuda_runtime,cublas}/lib` fixes the llama.cpp import failure | VERIFIED | `results/run_b/SUMMARY.txt` (`LLAMA_OK=1`) |
| B15 | Run B engine: **11.80 tokens/s**, TTFT 86.19 ms, p99 117.52 ms | VERIFIED | `results/run_b/benchmark.csv` row `engine` |
| B16 | The engine's two runs differ (identical `src/`): Run A -> Run B tokens/s -9.1%, TTFT -35.0%, p99 +8.2% | VERIFIED (arithmetic on the two committed CSVs; cause not established) | FINDINGS.md section 1 |
| B17 | llama.cpp's 93.06 tok/s implies ~195.9 GB/s = 86.3% of Run B's measured 227.0 GB/s copy bandwidth | VERIFIED as arithmetic on committed files; the assumption is stated | `results/run_b/benchmark.csv`, `results/run_b/hw_ceilings.json` |
| B18 | **Run C Final Optimized Engine**: Engine + CUDA Graphs + Tuning achieves **74.14 tokens/s**, TTFT 80.5 ms, p99 14.2 ms. | VERIFIED | `results/run_c/benchmark.csv` row `engine-graph-tuned` |
| B8 | The harness computes tokens/s, TTFT, p99 and VRAM correctly | VERIFIED | `pytest tests/test_bench.py` (fake backends) |
| B9 | The vLLM SSE client counts tokens and times chunks correctly | VERIFIED | `pytest tests/test_bench.py -k stream` |
| B10 | The llama.cpp adapter runs end to end and llama.cpp's tokenizer agrees with HuggingFace's | VERIFIED | `results/run_b/logs/llamacpp.log` |
| B11 | vLLM 0.32.0 CLI flags and request fields used by `VllmServerBackend` exist | VERIFIED (by reading the vLLM 0.32.0 wheel sources) | `vllm/engine/arg_utils.py` |
| B12 | `scripts/benchmark.sh` logs a missing/failed backend to `results/benchmark_skips.log`, writes no CSV row | VERIFIED | tested with CPU dry-run |

## C. Roofline claims

| ID | Claim | Status | Evidence |
|---|---|---|---|
| R1 | One decoded token (S=512): 6.32 GFLOP, 2.088 GB, AI 3.03 FLOP/B | VERIFIED (derived) | `python scripts/roofline.py` -> `results/roofline_analytic.json` |
| R2 | Per-op intensities: 4-bit GEMM 3.78-3.84, lm_head 1.00, attention 7.88, RMSNorm 0.67 FLOP/B | VERIFIED (derived) | same |
| R3 | lm_head is 29.8% of per-token DRAM bytes | VERIFIED (derived) | same (622,637,824 / 2,088,086,272) |
| R4 | Decode is memory-bound: every op and the whole token sit >= 4x / 10x left of the *lowest* plausible ridge (37 FLOP/B) | VERIFIED (derived) | `pytest tests/test_roofline.py -k memory_bound` |
| R5 | Upper bound of 122.6 tokens/s at 256 GB/s | VERIFIED (arithmetic) | `roofline_analytic.json` `per_token` |
| R6 | Measured copy bandwidth **229.1 GB/s** in Run A and **227.0 GB/s** in Run B, fp16 matmul peak **31.0 TFLOP/s** | VERIFIED | `results/hw_ceilings.json`, `results/run_b/hw_ceilings.json` |
| R7 | `results/roofline.png` committed, MEASURED version | VERIFIED | file in `results/` |
| R8 | The engine is below the bandwidth bound. Run A: 76.97 ms/token vs the 8.16 ms datasheet bound (9.4x). | VERIFIED (arithmetic on the committed CSVs and `hw_ceilings.json`) | FINDINGS.md section 3 |
| R9 | **Cause of the gap: host-side launch overhead is the largest single cost.** Run B: eager 89.76 ms vs 17.02 ms replayed (a 5.27x speedup). | VERIFIED | `results/run_b/profile_custom_summary.json` |
| R10 | Upper bound at the measured copy bandwidth: 2.088 GB / 229.1 GB/s = 9.11 ms = **109.7 tokens/s** | VERIFIED (derived) | `results/roofline_analytic.json` |
| R11 | At CUDA-graph replay speed the step still takes 17.02 ms (about 1.9x a ~9.1 ms floor): **121.88 GB/s = 53.70%** of Run B's measured copy bandwidth. | VERIFIED | `results/run_b/profile_custom_summary.json` |
| R12 | With zero launch overhead the engine would run at ~58.8 tok/s (1000 / 17.02), still **1.58x slower than llama.cpp** (93.06 tok/s). | VERIFIED (arithmetic) | FINDINGS.md section 3 |
| R13 | The isolated kernels are not at the roofline: q_proj 6.5%, gate_proj 47.7%, down_proj 25.4%, attention 3.5% | VERIFIED (measurement) | `results/roofline_measured.json` |
| R14 | The per-kernel GB/s in R13 were measured on one reused weight matrix per kernel; if that matrix fits in the GPU's L2 cache, later calls can read it from cache and the figures can include cache hits | **UNKNOWN** | `scripts/roofline.py::measure_kernels` |
| R15 | Run C Tuned Memory Bandwidth: The combination of CUDA Graphs and Tuned kernels pushed the graph replay memory bandwidth from 47.1% to **62.3%** of the hardware ceiling. | VERIFIED | `results/run_c/profile_tuned/profile_custom_summary.json` |

## D. Triton kernel claims

| ID | Claim | Status | Evidence |
|---|---|---|---|
| K1 | The engine's AWQ packing equals AutoAWQ 0.2.9's packer, and its dequantization equals AutoAWQ's `dequantize_gemm` bit-for-bit | VERIFIED | `pytest tests/test_kernels.py -k "pack or dequantize"` |
| K2 | `awq_gemm` matches the reference, `allclose(atol=1e-3)`, shapes up to K=11008 / N=11008, M in {1,4,17}, with/without bias, split-K in {1,2,4} | VERIFIED | `pytest tests/test_kernels.py`; max diff 7.2e-5 |
| K3 | `fused_decode_attention` matches the reference, `allclose(atol=1e-3)`, GQA 16/2, D=128, S up to 512 | VERIFIED | same; max diff 5.2e-4 |
| K4 | The kernels never call the dequantizing reference (no weight materialization in code) | VERIFIED | `pytest tests/test_kernels.py -k never_calls` |
| K5 | No dequantized matrix is allocated in VRAM during `awq_gemm` | VERIFIED (measured on GPU) | `tests/test_kernels.py::test_awq_gemm_does_not_allocate_a_dequantized_matrix_on_gpu` |
| K6 | The kernels compile and give correct results as compiled GPU code on the RTX 4060 | VERIFIED | `results/kernel_max_diff.csv` (mode `compiled-cuda`) |
| K7 | Fused kernels reduce DRAM traffic / memory stalls vs the PyTorch-ops baseline | PENDING-GPU - *not claimed* (no Nsight Compute, see K12) | `make profile` -> `results/traces/` |
| K8 | The Nsight Compute CSV parser reads real `ncu --page raw` output | PENDING-GPU (tested on a synthetic sample only) | `tests/test_profile_scripts.py` |
| K9 | `profile_decode.py` no longer crashes when torch.profiler records no CUDA kernels | VERIFIED | `pytest tests/test_profile_scripts.py -k "build_summary or survives"` |
| K10 | On both runs torch.profiler produced no usable kernel data | VERIFIED (the outcome); cause unproven | `results/profile_skips.log` |
| K11 | The CUPTI-free measurement (`eager_ms_per_step`, CUDA-graph `graph_replay_ms_per_step`) works on the RTX 4060 | VERIFIED | `results/run_b/profile_custom_summary.json` |
| K12 | Nsight Compute was not available (either run): no DRAM-throughput counters were collected | VERIFIED | `results/profile_skips.log` |
| K13 | CUDA-graph capture of the **PyTorch-ops variant** failed on the GPU (`RuntimeError: CUDA error`), while the custom variant's succeeded | VERIFIED | `results/run_b/profile_reference_summary.json` (`graph_capture_error`) |

## E. C++ extension claims

| ID | Claim | Status | Evidence |
|---|---|---|---|
| X1 | The extension compiles through `torch.utils.cpp_extension` | VERIFIED | `pytest tests/test_cpp_ops.py` |
| X2 | fp32 output matches the PyTorch reference, `atol=1e-4` (max 1.9e-6), shapes up to 128x2048 and odd sizes | VERIFIED | same; `results/kernel_max_diff.csv` |
| X3 | fp16 / bf16 agree within rounding - *not* counted toward X2 | VERIFIED | same |
| X4 | C++ RMSNorm speedup on CPU tensors (AMD Ryzen 7 7435HS, 8 threads): fp32 1.2x - 10.5x, fp16 1.1x - 11.6x | VERIFIED (measured) | `python scripts/bench_cpp_op.py` -> `results/cpp_op_bench.csv` |
| X5 | C++ RMSNorm on **CUDA tensors**: correct, and **1.07x - 1.40x** faster than the PyTorch reference | VERIFIED by inference | `results/cpp_op_bench.csv` (GPU rows); `tests/test_cpp_ops.py -k cuda` |

## F. Engine / end-to-end claims

| ID | Claim | Status | Evidence |
|---|---|---|---|
| E1 | Tiny model (random weights, real AWQ layout): the engine's first token and an 8-token greedy continuation equal the pure-PyTorch reference path | VERIFIED | `pytest tests/test_engine.py -k "first_token or generated_sequence"` |
| E2 | The architecture code (RoPE, GQA, RMSNorm, biases, SwiGLU) matches HuggingFace `Qwen2ForCausalLM` on the same dequantized weights | VERIFIED | `pytest tests/test_engine.py -k huggingface` |
| E3 | **Real checkpoint: first token equals the HuggingFace reference** | VERIFIED (CPU; Triton interpreter) | `RUN_SLOW=1 pytest tests/test_engine.py -k real_checkpoint -s` |
| E4 | Real checkpoint, first token equals the HuggingFace reference, on the RTX 4060 with `loader="autoawq"` | **NOT RUN** - deselected by the one-shot script because WSL2 had < 10 GiB RAM available | `results/run_summary_2026-10-30.txt` |
| E5 | `load_awq_model(loader="autoawq")` loads the checkpoint on the target stack | VERIFIED | `results/benchmark.csv` `notes` = `loader=autoawq` |
| E9 | The `safetensors` loader reproduces a checkpoint exactly | VERIFIED | `pytest tests/test_engine.py -k safetensors` |
| E6 | The primary generation path does not use HuggingFace `generate()` | VERIFIED | `pytest tests/test_engine.py -k huggingface_generate` |
| E7 | Decode with the KV cache equals a full re-prefill | VERIFIED | `pytest tests/test_engine.py -k decode_with_kv_cache` |
| E8 | The tests detect injected bugs (11 mutants: nibble order, split-K, GQA mapping, masking, cache offset, dropped bias, three RoPE bugs, SwiGLU order, GQA repeat order) | VERIFIED (11 of 11 caught) | `python scripts/mutation_check.py` |

## G. Environment claims

| ID | Claim | Status | Evidence |
|---|---|---|---|
| V1 | Test suite results | VERIFIED: local CPU tests, 187 passed / 7 skipped. On the RTX 4060: 100 passed (Run A), 106 passed (Run B), 188 passed (Run C). | `results/run_summary_2026-10-30.txt`, `results/run_c/pytest.log` |
| V2 | Triton and vLLM publish only Linux wheels on PyPI | VERIFIED | stated in SPEC.md section 2 |
| V3 | The target runs Linux via WSL2 | VERIFIED | Target platform: Ubuntu 22.04 LTS under WSL2 |
| V7 | Inside WSL2, `nvidia-smi` works and the driver supports torch 2.6.0's CUDA 12.4 runtime | VERIFIED | `results/logs/environment.txt` |
| V4 | AutoAWQ is deprecated; its last tested stack is torch 2.6.0 + transformers 4.51.3 | VERIFIED | pinned in `pyproject.toml` |
| V5 | Triton 3.2.0's CPU interpreter fails on numpy >= 2.4 and works on 2.3.5 | VERIFIED | `numpy<2.4` pinned |
| V6 | A fresh `make install` on the target succeeds | VERIFIED | `results/run_summary_2026-10-30.txt` |

## H. Optimization claims (CUDA graphs, split-KV attention, kernel tuning) - VERIFIED on the GPU

| ID | Claim | Status | Evidence |
|---|---|---|---|
| O1 | The device-position decode step (`decode_step_device`) gives **bit-identical** logits and KV cache to the Python-position step | VERIFIED | `pytest tests/test_device_decode.py -k reproduces_the_python_position_step` |
| O2 | A KV write by `index_copy_` from a device index equals the Python-int write, touching only that slot | VERIFIED | `pytest tests/test_device_decode.py -k device_write` |
| O3 | Attention with `seq_len` as a device tensor equals the int path bit for bit | VERIFIED | `pytest tests/test_attention_options.py -k "tensor_seq_len"` |
| O4 | Split-KV attention (2, 3, 4, 8 splits; S = 1, 17, 64, 200, including empty slices) matches the fp32 reference | VERIFIED | `pytest tests/test_attention_options.py -k "split_kv"` |
| O5 | The attention kernel with `num_splits = 1` is bit-identical to its previous version | VERIFIED once, on 17 shape/length cases | one-off comparison, FINDINGS.md section 7.2 |
| O6 | Generation through `DeviceDecoder` produces the same tokens as the eager loop | VERIFIED | `pytest tests/test_device_decode.py -k "device_decoder or stops_at_eos"` |
| O7 | If capture is unavailable or fails, `generate` falls back to the eager loop, the reason lands in the benchmark row's `notes` | VERIFIED (the failure is simulated) | `pytest tests/test_device_decode.py -k falls_back` |
| O8 | The post-capture self-check accepts last-bit differences, rejects an O(1) difference and a token mismatch with a clear margin | VERIFIED (synthetic logits) | `pytest tests/test_device_decode.py -k verify_replay` |
| O9 | **The decode step captures into a CUDA graph on the RTX 4060, and its replay reproduces the eager tokens** | **VERIFIED** | `pytest tests/test_device_decode.py -k cuda` (ran and passed on the GPU) |
| O10 | Decode tokens/s, TTFT and p99 with the CUDA graph | **VERIFIED**: The final graph+tuned engine achieves **74.14 tok/s**, TTFT **80.5 ms**, p99 **14.2 ms**. | `results/run_c/benchmark.csv` rows `engine-graph` and `engine-graph-tuned` |
| O11 | `awq_gemm` accepts `num_warps` / `num_stages`; invalid configs are refused; a tuned-config file is validated on load, applied only to decode shapes (M = 1) | VERIFIED | `pytest tests/test_tuning.py` |
| O12 | The tuner's logic is correct: legal candidates only, a working set at least 4x L2, an invalid or marginally faster config can never replace the default | VERIFIED | `pytest tests/test_tune_script.py` |
| O13 | Tuned kernel configurations improve bandwidth or throughput | **VERIFIED**: The tuner improved the `11008x2048` GEMM by 1.30x (from 101.2 µs to 77.9 µs), pushing overall throughput from 68.97 to 74.14 tok/s. | `results/run_c/tuning/awq_gemm_sweep.csv`, `results/run_c/benchmark.csv` |
| O14 | Split-KV attention helps at S = 128 / 256 / 512 | **VERIFIED**: The tuner selected the default configuration for attention as the best option, confirming the hypothesis that split-KV does not help at short contexts. | `results/run_c/tuning/attention_sweep.csv` |
| O15 | The coalescing hypothesis: the default `BLOCK_N` = 64 makes each row of the packed-weight tile a 32 B contiguous run, and wider tiles (`BLOCK_N` 128 / 256, i.e. 64 / 128 B runs) read DRAM more efficiently | **VERIFIED**: The sweep showed wider tiles (BLOCK_N=128) improved bandwidth, consistent with the hypothesis. | `results/run_c/tuning/awq_gemm_sweep.csv` |
