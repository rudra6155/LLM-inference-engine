# LLM Inference Engine (From Scratch)

A batch-1 LLM inference engine built from scratch in PyTorch, Triton, and C++, running 4-bit AWQ quantized Qwen2.5-3B on an 8GB RTX 4060 Laptop GPU. 

No vLLM, no HuggingFace `generate()`, no standard PyTorch attention modules—just hand-rolled Triton kernels, a custom C++ RMSNorm extension, explicit KV cache management, and CUDA graph replay.

The core motivation was first-principles systems profiling: build an autoregressive engine from scratch, find out why naive eager execution gets decimated on consumer hardware, and optimize it to the hardware memory bandwidth ceiling.

---

## Benchmark Results (RTX 4060 Laptop GPU, 8GB GDDR6)

Starting from naive eager execution at **12.81 tokens/sec**, I profiled the host/device bottleneck and pushed throughput to **74.14 tokens/sec** via CUDA graph capture and Triton kernel tuning.

| Backend | Decode Tokens/sec | TTFT (ms) | Inter-Token p99 (ms) | Peak VRAM (MiB) | Bandwidth Utilization* |
|---|---|---|---|---|---|
| **Custom Engine (Eager)** | 12.81 | 132.2 | 108.0 | 2823 | 11.6% |
| **Custom Engine + CUDA Graphs** | 68.97 | 78.8 | 15.3 | 2897 | 62.5% |
| **Custom Engine + Graphs + Tuned Kernels** | **74.14** | **80.5** | **14.2** | **2897** | **67.2%** |
| **llama.cpp (Q4_K_M)** | 101.70 | 34.8 | 11.7 | 2285 | 86.3% |

*Model: Qwen2.5-3B-Instruct (4-bit AWQ / GGUF Q4_K_M), context 128 prompt + 128 decode, batch size 1. Measured on Ubuntu 22.04 LTS via WSL2.*
*\*Bandwidth utilization relative to measured device copy bandwidth (229.1 GB/s). Relative to the 256 GB/s datasheet ceiling, the tuned engine achieves 60.1% (isolated profile trace in `profile_custom_summary.json` measured 62.3% at 14.93ms step time on an unboosted run).*

The engine doesn't beat llama.cpp—it trails by **1.37x** (101.70 / 74.14). But profiling explained exactly where every microsecond goes: closing the gap from **7.9x slower (Run C baseline) down to 1.37x** by eliminating Python launch overhead and fixing DRAM access coalescing. Full analysis in [`FINDINGS.md`](FINDINGS.md).

---

## Architectural Breakdown

1. **Fused Dequantize + GEMM Triton Kernel (`awq_gemm`):**
   - Autoregressive decode is strictly memory-bandwidth bound (arithmetic intensity = 3.03 FLOP/B for Qwen2.5-3B at batch size 1).
   - Dequantizes 4-bit AWQ packed weights (int32 holding 8 int4 weights) directly in SRAM during the matrix multiplication.
   - Zero intermediate fp16 weight tensor allocation in VRAM.

2. **Fused Decode Attention Triton Kernel (`fused_decode_attention`):**
   - FlashAttention-style causal single-token decode attention with Grouped-Query Attention (GQA: 16 query heads, 2 KV heads).
   - Keeps softmax online reductions in registers, reading keys and values directly from the preallocated KV cache.

3. **C++ RMSNorm Extension (`src/csrc/custom_ops.cpp`):**
   - Custom PyTorch C++ extension compiled via `torch.utils.cpp_extension`.
   - CPU path: Single-pass hand-written loop with AVX2/F16C hardware conversion intrinsics (10x speedup over PyTorch eager at batch 1).
   - CUDA path: Dispatches ATen kernels inside C++ to shave off Python interpreter overhead per sub-layer.

4. **CUDA Graph Capture & Replay:**
   - Eager PyTorch was spending **~81% of execution time in host launch overhead** (profiler measured ~82–90 ms eager step vs ~15–17 ms replayed via CUDA graph). Dispatching ~350+ individual operations per token through the Python runtime and CUDA driver queues completely dwarfed actual GPU execution time.
   - The entire decode iteration is captured into a single CUDA graph and replayed with a single launch.
   - Solved the `.item()` synchronization trap by maintaining token position offsets directly on device tensors (`index_copy_` and `index_select`).

5. **Hardware Roofline & Automated Kernel Tuner:**
   - Measures actual device copy bandwidth (measured: **229.1 GB/s** vs 256 GB/s datasheet ceiling).
   - `scripts/tune_kernels.py` sweeps `BLOCK_N`, `BLOCK_K`, warp count, pipeline stages, and `SPLIT_K` across the model's projection shapes (`2048x256`, `2048x2048`, `2048x11008`, `11008x2048`).
   - Switching from default 64-width blocks to `BLOCK_N=128, BLOCK_K=128` improved DRAM coalescing on the 4060, yielding a 1.30x speedup on the MLP down-projection.

---

## Quick Start (Linux / WSL2)

Prerequisites: NVIDIA GPU with CUDA 12.x support (tested on RTX 4060 Laptop GPU), Python 3.10.

```bash
# 1. Clone repository
git clone https://github.com/rudra6155/llm-inference-engine.git
cd llm-inference-engine

# 2. Install dependencies and build C++ extension
make install

# 3. Run test suite (verifies kernel numerics and KV cache math)
make test

# 4. Run full benchmark suite (Custom Engine vs baselines)
make benchmark

# 5. Measure hardware ceilings and plot roofline chart
make roofline

# 6. Run GPU kernel tuner
make tune
```

---

## What Broke & Lessons Learned

- **WSL2 CUPTI Failure:** `torch.profiler` and Nsight Compute silently recorded 0 kernel events due to known WSL2 CUPTI driver limitations. Diagnosed this by building a custom timing harness using `torch.cuda.Event`.
- **CUDA Graph Host-Tensor Freezing:** Passing integer token positions from Python froze the argument inside the graph, causing subsequent tokens to overwrite the same KV slot. Fixed by driving KV cache indexing via device tensors.
- **Reference Path Device-Side Assert:** Attempting to CUDA-graph capture PyTorch's eager reference ops crashed with device-side assertions due to dynamic memory allocations inside PyTorch's native forward pass.
- **Shared Memory Limits in Triton Sweeps:** Large block configurations (`BLOCK_N=256, BLOCK_K=128`) exceeded the 100 KB shared memory limit per SM on the RTX 4060. Added a hardware resource filter to the tuner to prune invalid launch grids.

---

## Why llama.cpp is Still 1.37x Faster

1. **Complete Kernel Fusion:** `llama.cpp` fuses QKV projection, RoPE embedding, KV cache insertion, multi-head attention, and output projection into a single unified CUDA kernel. My engine still dispatches separate Triton kernels for projections and attention.
2. **Zero Runtime Allocations:** `llama.cpp` uses raw `ggml` pointer offsets and static memory arenas. PyTorch, even within a CUDA graph, incurs small allocator bookkeeping overhead.
3. **Pure C++ Runtime:** `llama.cpp` is 100% native C/C++ without Python runtime dispatch.

---

## Technical Specifications & Claims

Detailed hardware analysis, analytic memory derivation, and empirical verification data are documented in:
- [`FINDINGS.md`](FINDINGS.md) — Comprehensive engineering log, profiling breakdowns, and failure analyses.
- [`SPEC.md`](SPEC.md) — Architectural specifications, model parameters, and scoping decisions.
- [`CLAIMS.md`](CLAIMS.md) — Empirical verification matrix matching every performance claim against raw CSV artifacts in `results/`.
