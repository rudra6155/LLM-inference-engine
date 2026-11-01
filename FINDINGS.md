# Findings & Engineering Log

This document tracks the process of building this engine from scratch: what broke, what was measured, and why the final numbers look the way they do. 

All raw data is committed in the `results/` folder across three distinct runs: **Run A** (baseline), **Run B** (profiling), and **Run C** (the final optimized engine). 

---

## 1. The Baseline & The Launch Overhead Tax (Run A & B)

When I first got the engine running with eager PyTorch execution, I hit **12.81 tokens/sec** with a Time-To-First-Token (TTFT) of 132ms. For an 8GB RTX 4060, this felt far too slow.

I ran a roofline analysis. For a 4-bit Qwen2.5-3B model at batch size 1, we have to read 2.074–2.088 GB of weights from VRAM to generate one token. The arithmetic intensity is 3.03 FLOP/B, which means the operation is heavily memory-bound. The RTX 4060 has a measured memory copy bandwidth of ~229 GB/s (datasheet ceiling: 256 GB/s). Therefore, the theoretical floor for token generation was about 9.1ms per token (~109 tokens/sec). Eager execution was at 78.1ms per token (12.81 tok/s)—roughly 8.6x off the theoretical limit.

To find out why, I tried to profile using `torch.profiler` and Nsight Compute. **This completely failed.** CUPTI is fundamentally broken under WSL2, so it recorded zero kernel events. 

Instead, I wrote a custom timing harness using CUDA events. By comparing eager execution against CUDA graph replays, I measured an eager step time of ~82–90ms vs an actual GPU execution time of ~15–17ms. The other ~67–73ms was pure host-side dispatch and launch overhead. Dispatching ~350+ individual operations per token (GEMMs, attention, RMSNorms, RoPE) through the Python interpreter and CUDA runtime queues meant Python was spending ~81% of its time just telling the GPU what to do.

*(Note: llama.cpp on the same machine achieved 101.70 tok/s. My eager engine was 7.94x slower).*

---

## 2. The Biggest Win: CUDA Graphs (Run C)

To eliminate the Python launch overhead, I implemented CUDA graphs. The entire decode step is captured into a single graph and replayed. 

This was the single most effective optimization. Throughput jumped from **12.81 to 68.97 tok/s** (a 5.4x speedup). The p99 latency dropped from 108ms to 15.3ms.

**What broke during graph capture:**
*   **The `.item()` trap:** The initial engine passed the token position as a Python integer. A graph freezes all host arguments, so replaying it would have reused one position forever. I had to rewrite the decode step to pass the position as a device tensor, using `index_copy_` for the KV cache and `index_select` for RoPE.
*   **The PyTorch-ops failure:** Capturing the reference PyTorch ops threw a `device-side assert triggered`. PyTorch dynamically allocates memory during the forward pass, which is illegal during graph capture. I had to fall back to eager execution for the reference path (which is why the ablation is so slow).
*   **Shared Memory OOMs:** The tuner kept crashing with `OutOfResources` errors. I wrote a check that calculates the required shared memory for each block config and skips it if it exceeds the 4060's 101KB limit.

Even without the launch overhead, the step took 17.02ms. The kernels themselves were still 1.9x away from the memory bandwidth roofline.

---

## 3. Kernel Tuning (Run C)

To fix the memory bandwidth efficiency, I wrote an automated tuner (`scripts/tune_kernels.py`). It sweeps `BLOCK_N`, `BLOCK_K`, `num_warps`, `num_stages`, and split-K configurations for the decode shapes. 

Two key findings emerged:
1.  **The default GEMM config was suboptimal for the 4060.** By switching to `BLOCK_N=128` and `BLOCK_K=128` with split-K=8, the `11008x2048` GEMM improved by **1.30x** (from 101.2 µs down to 77.9 µs). The wider tiles increased contiguous memory transactions, improving DRAM efficiency.
2.  **Split-KV attention does not help at short contexts.** Sweeping splits of 2, 4, and 8 for sequence lengths up to 512 showed that the default split-1 configuration was consistently fastest.

After tuning, final throughput hit **74.14 tokens/sec**. Memory bandwidth utilization reached **62.3%** of measured copy bandwidth under the isolated profiler trace (14.93ms step time), and **67.2%** in the end-to-end benchmark (13.48ms step time, reading ~153.8 GB/s vs 229.1 GB/s measured copy bandwidth, or 60.1% of the 256 GB/s datasheet ceiling). 

---

## 4. The Final Numbers (Run C) & The Remaining Gap

| Backend | Tokens/sec | TTFT (ms) | ITL p99 (ms) | VRAM (MiB) |
|---|---|---|---|---|
| Custom Engine (Eager) | 12.81 | 132.2 | 108.0 | 2823 |
| Custom Engine + CUDA Graphs | 68.97 | 78.8 | 15.3 | 2897 |
| **Custom Engine + Graphs + Tuning** | **74.14** | **80.5** | **14.2** | **2897** |
| llama.cpp (Q4_K_M) | 101.70 | 34.8 | 11.7 | 2285 |

The engine is **1.37x slower** than llama.cpp on the exact same hardware. Why?

*   **Fused Attention:** llama.cpp has a hand-written CUDA kernel that fuses the entire attention block (QKV projection, RoPE, KV cache read, softmax, and output projection) into one operation. My engine uses separate Triton kernels for each.
*   **No PyTorch Allocations:** Even inside a CUDA graph, PyTorch's tensor allocator incurs background bookkeeping. llama.cpp uses raw `ggml` pointers and memory pools.
*   **C++ Execution:** llama.cpp is 100% native C++. My engine still has a Python front-end managing the generation loop and tokenizer.

---

## 5. Memory Model & Roofline

**KV Cache:** For each generated token, each layer stores one K vector and one V vector per KV head:
`bytes per token = 2 (K and V) * num_layers (36) * num_kv_heads (2) * head_dim (128) * dtype_bytes (2) = 36,864 bytes (36 KiB)`.
At the default `max_seq_len` of 1,024, this is 36 MiB. Weights held once (tied lm_head) are 1.92 GiB. The 8GB card is therefore not capacity-constrained for this model; it is purely bandwidth-bound.

**Roofline:** Measured copy bandwidth on the RTX 4060 Laptop is **229.1 GB/s** (89.5% of the 256 GB/s datasheet figure). At this bandwidth, the theoretical floor for reading 2.088 GB per token is 9.11 ms (109.7 tok/s). 
At graph speed, execution reached **121.9 GB/s = 53.7%** of measured copy bandwidth. Kernel tuning brought this up to **62.3%**.

---

## 6. The C++ Extension (RMSNorm)

`src/csrc/custom_ops.cpp` reproduces HuggingFace `Qwen2RMSNorm` numerics (fp32 statistics, normalized value rounded to activation dtype, multiplied by weight).

*   **On CPU:** Hand-written kernel. Achieves a 10x speedup at 1 row, shrinking to 1.7x at 128 rows.
*   **On CUDA:** Composed from ATen ops inside C++. This removes Python dispatch overhead per call but launches the same ATen kernels as the reference. Speedup is a modest 1.07x - 1.40x.

---

## 7. Things That Broke (And How I Fixed Them)

*   **CUPTI is broken in WSL2:** `torch.profiler` returned empty traces. Fixed by writing a custom timing harness using `torch.cuda.Event`.
*   **CUDA Graph Capture Failure:** Capturing the PyTorch reference ops threw a `device-side assert triggered`. Fell back to eager execution for the reference ablation path.
*   **VRAM Drift:** Peak VRAM jumped from 2,899 MiB in Run A to 3,703 MiB in Run B. Investigated and discovered NVML-based monitoring reads device-level VRAM, and desktop background processes were allocating memory between runs.
*   **C++ op slower than PyTorch on initial build:** Early build without hardware fp16 conversion was slower than PyTorch for fp16 on CPU (0.14x speedup). Fixed by adding the F16C intrinsic path.
*   **Triton 3.2.0 CPU interpreter compatibility:** Fails on `numpy >= 2.4`. Pinned `numpy<2.4` for reproducible CPU unit tests.

---

## 8. What's Next?

To close the remaining 1.37x gap to llama.cpp:
1.  Write a fully fused attention kernel in Triton that merges QKV projection and RoPE.
2.  Port the generation loop to C++ using `libtorch`.
3.  Implement continuous batching with a paged KV cache.
4.  Quantize `lm_head` to 4-bit (currently in fp16, accounting for ~30% of per-token DRAM traffic).
