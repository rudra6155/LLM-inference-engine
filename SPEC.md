# SPEC — LLM inference engine (batch size 1, 4-bit AWQ, RTX 4060)

This file states the scope **before** the code. Anything not stated here is not
promised. Status of every claim lives in `CLAIMS.md`.

## 1. Mission

A small autoregressive inference engine for a ~3B-parameter model in 4-bit
quantization on an 8 GB laptop GPU, whose credibility rests on four things:

1. custom **Triton kernels** for the memory-bound decode phase (fused
   dequantize+GEMM, fused decode attention),
2. a **roofline analysis** showing why decode is memory-bandwidth-bound,
3. a **C++ extension** for one hot-path operation (RMSNorm),
4. a **benchmark** (tokens/s, TTFT, p99 latency, peak VRAM) against vLLM and
   llama.cpp, with a profiler trace.

## 2. Target hardware and platform

| Item | Value | Source / status |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB GDDR6 | Target hardware |
| CUDA | 12.x / 12.4 | System CUDA runtime |
| Memory bandwidth | 256 GB/s theoretical (128-bit bus x 16 Gbps effective) | Datasheet specification; **measured at run time** by `scripts/roofline.py` |
| Peak FLOPs | depends on the laptop's power limit (TGP) | **not assumed**; measured at run time by `scripts/roofline.py` |
| OS | **Linux via WSL2** | Ubuntu 22.04 LTS under WSL2. Required for Triton and CUDA 12.4 manylinux wheels. |

Development was split between a local CPU development setup (using the Triton interpreter for kernel numerical verification) and the RTX 4060 Laptop GPU for CUDA graph capture, kernel profiling, and benchmarking.

## 3. Model

* `Qwen/Qwen2.5-3B-Instruct-AWQ` (single `model.safetensors`), 4-bit AWQ,
  `group_size=128`, `zero_point=true`, `version=gemm`.
* Architecture values read from the checkpoint's `config.json`:
  `hidden_size=2048`, `intermediate_size=11008`, `num_hidden_layers=36`,
  `num_attention_heads=16`, `num_key_value_heads=2`, `vocab_size=151936`,
  `rms_norm_eps=1e-6`, `rope_theta=1e6`, `tie_word_embeddings=true`,
  `torch_dtype=float16`. `head_dim` is not a config key; it is
  `hidden_size / num_attention_heads = 128`.
* q/k/v projections carry a bias; o_proj and the MLP projections do not
  (read from the checkpoint's tensor index).
* Activations and KV cache are fp16.

## 4. Baselines

| Baseline | Model file | Notes |
|---|---|---|
| vLLM 0.32.0 (OpenAI-compatible server, streamed) | the same AWQ checkpoint | VRAM is policy-dependent (`--gpu-memory-utilization` pre-allocates KV cache). If vLLM cannot be installed, the benchmark falls back to llama.cpp only and `FINDINGS.md` says so under Limitations. |
| llama.cpp via `llama-cpp-python` 0.3.36 | `Qwen/Qwen2.5-3B-Instruct-GGUF`, `qwen2.5-3b-instruct-q4_k_m.gguf` | **Different 4-bit format** (GGUF Q4_K_M, not AWQ g128). Comparison is "same model, best-available 4-bit file per runtime", not bit-identical weights. |

All backends run in separate processes, receive the **same prompt token ids**,
use greedy decoding, ignore EOS, and generate a fixed number of new tokens.

## 5. Metric definitions (identical for every backend)

* **TTFT**: wall time from submitting the prompt to the first generated token
  being available to the calling process (includes prefill).
* **Decode tokens/s**: `(n_generated - 1) / (t_last_token - t_first_token)`,
  pooled over all timed runs.
* **p99 latency**: 99th percentile of the inter-token interval (time between
  successive tokens becoming available to the calling process), pooled over all
  tokens of all timed runs. Includes Python / launch / HTTP overhead as seen by
  the caller.
* **Peak VRAM**: peak *device-level* used memory (NVML) during load plus
  generation, minus the used memory measured before the backend was loaded.
  Device-level is used because vLLM runs the model in a separate process.

## 6. Target (falsifiable)

At batch size 1, the engine's decode tokens/s on the RTX 4060 is higher than
that of each baseline that could be run. If it is not, `FINDINGS.md` reports it
as **not met** and the profile-then-optimize protocol applies. No claim is made
until `results/benchmark.csv` contains the numbers.

## 7. Scope

**In scope:** AWQ 4-bit loading via `autoawq`; manual KV cache; greedy
autoregressive loop; Triton fused dequant+GEMM; Triton fused decode attention
(batch 1, short sequences, GQA); C++ RMSNorm; benchmark; profiling; roofline;
KV-cache memory model from first principles.

**Out of scope:** training, continuous batching, multi-GPU, tensor parallelism,
sampling strategies other than greedy, batch size > 1, projection
fusion (QKV / gate-up). CUDA graphs and split-KV attention were out of scope
until a profile motivated them; the Run B profile did, and they were built on
2026-10-30 (section 10, item 8). Fusion remains an unbuilt candidate.

## 8. Requirements & Verification Matrix

| # | Requirement | Test / artifact | Verifiable without GPU? |
|---|---|---|---|
| 1a | Load 4-bit model with `autoawq` | `engine.model.load_awq_model(..., loader="autoawq")` | **No** (needs CUDA + autoawq kernels path) |
| 1b | KV bytes/token = `2 * L * H_kv * D * bytes` | `tests/test_engine.py` (formula + allocated tensor size) | Yes |
| 1c | First generated token equals reference exactly | `tests/test_engine.py` (tiny model, interpreter) and real-checkpoint test | Tiny model: yes. Real checkpoint: on CPU only with `RUN_SLOW=1` (very slow); on GPU by default |
| 2a | Fused dequant+GEMM Triton kernel, no materialized weights, `allclose(atol=1e-3)` | `tests/test_kernels.py` | Correctness: yes, with `TRITON_INTERPRET=1` (CPU). **Compiled GPU run: no** |
| 2b | Fused decode attention Triton kernel, `allclose(atol=1e-3)` | `tests/test_kernels.py` | same as 2a |
| 2c | Roofline plot | `results/roofline.png` | Analytic version: yes. Measured points/ceilings: no |
| 3 | C++ RMSNorm via `torch.utils.cpp_extension`, `atol=1e-4` | `tests/test_cpp_ops.py` | Yes (CPU path). CUDA-tensor path: no |
| 4a | Benchmark tokens/s, TTFT, p99, VRAM vs vLLM and llama.cpp | `scripts/benchmark.sh` -> `results/benchmark.csv` | **No** |
| 4b | Profile decode, DRAM-throughput trace, fused vs baseline | `scripts/profile.sh` | **No** |
| 5 | Benchmark script emits the four metrics | `tests/test_bench.py` (fake backend) | Harness logic: yes. Real numbers: no |

`TRITON_INTERPRET=1` runs Triton kernels as Python/numpy on the CPU. It
validates kernel **logic and numerics**; performance and compiled code behavior
are measured on the GPU.

## 9. Verification & Benchmarking Rules

* All GPU throughput, latency, VRAM, bandwidth-utilization, and speedup numbers
  in this repository are directly measured on the physical RTX 4060 Laptop GPU.
* CPU unit tests run through PyTorch references or `TRITON_INTERPRET=1` to validate
  kernel numerics and algorithm logic independently of GPU availability.
* Benchmark runs output raw artifacts into `results/` (CSV logs, JSON summaries,
  and profile plots).

## 10. Decisions and Technical Rationale

1. **AutoAWQ dependency:** Its upstream banner notes that the last tested stack
   is torch 2.6.0 + transformers 4.51.3. AutoAWQ is used as the primary loader;
   the pinned stack in `pyproject.toml` follows that verified configuration. A
   direct-safetensors loader is provided for CPU tests and as an explicit fallback.
2. **Baselines in separate virtualenvs:** vLLM 0.32.0 requires
   `transformers>=5.16.1`, which conflicts with the AutoAWQ pin.
   `scripts/benchmark.sh` isolates baselines using `VLLM_PYTHON` and `LLAMACPP_PYTHON`.
3. **C++ RMSNorm on CUDA tensors:** Composes ATen operations in C++ to eliminate
   Python dispatch overhead per call. On CPU tensors, it uses an explicit hand-written
   single-pass loop with F16C vector conversions.
4. **Tied embeddings:** The checkpoint stores `lm_head.weight` separately even
   though the config ties it; the engine ties them so the 622 MB matrix is held once.
5. **Tile sizes & tuning:** Kernel tile sizes and split-K are initialized with
   heuristics, then tuned empirically on the GPU.
6. **Loader Selection:** Keep `autoawq` as the default loader for standard 4-bit
   AWQ checkpoints, with `safetensors` available as an explicit fallback (`LOADER=safetensors`).
7. **Profile-driven optimization strategy:** Optimizations are implemented strictly
   based on profiling data. If per-step wall time is dominated by launch overhead,
   CUDA graphs are prioritized. If the GPU is compute/bandwidth-bound, kernel tile
   shapes, split-K, and warps are tuned.
8. **Iterative optimizations (Run C):** CUDA graphs, split-KV attention, and an
   automated kernel tuner (`scripts/tune_kernels.py`) were implemented to address
   the launch overhead and memory coalescing bottlenecks identified in Run B.
