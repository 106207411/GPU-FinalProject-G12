# EEP 524 Final Project Proposal: ResNet-18 Inference Engine on Tesla T4

## 1. Project Team Members — Team 12

| Name | Student ID | Primary role |
|---|---|---|
| Karry Cheng | 2680390 | Explicit/tiled SGEMM, register tiling, per-shape tuning |
| Andy Lee | 2577859 | Implicit GEMM, fused epilogue, FP16 tensor cores |
| Brian Su | 2680216 | Non-conv kernels, fusion, verification, profiling, submissions |

## 2. Tier 0 Baseline and Batch Size

**Baseline (unmodified starter):** **18.5 images/s on the Tesla T4** (course-measured Tier 0 figure, median 212.5 s per pass); we will reproduce it on the dev set with our T0 smoke-test submission on 11/22.

**Planned batch size: B = 64**, to be confirmed by a 32–128 sweep once T1 runs. Stage 4 has only 7×7 = 49 output pixels per image, so a small batch cannot fill the T4's 40 SMs; B = 64 gives every layer enough thread blocks while all buffers still fit easily in 16 GB.

## 3. Planned Tiers and Roadmap

**Goal:** T3 in the FP32 category by the freeze, every submission passing the check.py gate; T4 (FP16 category) is our stretch goal.

| Tier | Planned kernels / techniques | Owner | Target |
|---|---|---|---|
| T0 | Build starter, pass check.py, NCU + per-layer baseline, smoke-test submission | Brian | 11/22 |
| T1 | im2col + SMEM-tiled SGEMM, bank-conflict-free (padded) tiles, bias/ReLU fused in epilogue | Karry | 11/26 |
| T2 | Implicit GEMM (gather in tile load, no im2col buffer), residual add fused, per-shape launch configs | Andy | 12/2 |
| T3 | 2-D register tiling / thread coarsening on the T2 kernel, float4 loads (scalar fallback for K = 147), tile sizes tuned per shape class; split-K only if small layers stay under-filled | Karry | 12/9 |
| T4 | FP16 WMMA (16×16×16) / mma.sync m16n8k8, fp32 accumulation, fp16 weights + activations | Andy | Stretch, 12/14 |
| T5 | Stem conv + maxpool fusion, warp-shuffle avgpool and top-5, CUDA Graph for the forward pass | Brian | Stretch |

**Milestones:** by the design specification (**2 Dec**): T1 valid on the leaderboard and T2 passing check.py. By the leaderboard freeze (**15 Dec, 5 PM**): T3 submitted, plus T4/T5 if time allows.

## 4. Work Breakdown

| Member | Owns kernels | Other project tasks |
|---|---|---|
| Karry Cheng | T1 tiled SGEMM + fused bias/ReLU; T3 register-tiled implicit GEMM (built on Andy's T2 kernel), per-shape tile tuning, optional split-K | Roofline analysis, arithmetic-intensity predictions per shape class |
| Andy Lee | T2 implicit-GEMM gather + fused residual epilogue, per-shape launch configs; T4 WMMA / mma.sync kernel | FP16 accuracy check (per-layer ≤ 0.02 gate) |
| Brian Su | normalize, maxpool, avgpool, fc, top-5; T5 fusions and CUDA Graph | check.py regression after every change, NCU profiling, per-layer cudaEvent timing, submit.py submissions |
| All | Code review of each tier before submission | Design spec, presentation, final report |

## 5. Nsight Compute Metrics to Evaluate

**Protocol:** every tier is profiled on the T4 (Release, `-lineinfo`) at the planned batch on the dev set, capturing one complete batch (`--launch-count 25` for direct/implicit conv, 45 for explicit im2col, so the stage-4 layers, fc and top-5 are included), compared tier to tier. Debug set (B = 8) is for `check.py --layers`.

| Metric / section | Relevance to algorithm performance |
|---|---|
| `gpu__time_duration.sum` (+ cudaEvent per layer) | Hot layers / shape classes; tier speedup |
| `sm__throughput` / `dram__throughput .avg.pct_of_peak_sustained_elapsed` | Compute- vs memory-bound; did the tier move it |
| FLOPs (`sm__sass_thread_inst_executed_op_ffma_pred_on.sum` ×2) / `dram__bytes.sum` | Arithmetic intensity for roofline plots |
| `sm__warps_active.avg.pct_of_peak_sustained_active`, `launch__registers_per_thread`, `launch__waves_per_multiprocessor` | Occupancy vs ILP of register tiling; stage-4 fill |
| Warp State Statistics (stall long scoreboard / MIO throttle / barrier) | Latency hiding: DRAM, SMEM pressure, sync imbalance |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | SMEM tile conflicts; ~0 after padding/swizzle |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` / `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | Coalescing of NHWC gather / weights; ideal 4 (32-bit), 16 (float4) |
| `lts__t_sector_hit_rate.pct`, `dram__bytes_read.sum` | Data reuse; im2col vs implicit-GEMM DRAM traffic |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | T4 only: tensor-core utilization |

## 6. Primary References

- W. Hwu, D. Kirk, I. El Hajj, *Programming Massively Parallel Processors*, 4th ed., Morgan Kaufmann, 2022.
- K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognition," CVPR 2016.
- S. Chetlur et al., "cuDNN: Efficient Primitives for Deep Learning," arXiv:1410.0759, 2014 (implicit-GEMM convolution concept).
- S. Boehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance," 2022 (kernels 4–5 for register tiling).
- NVIDIA: CUDA C++ Programming Guide (WMMA), PTX ISA (mma.sync), Turing Tuning Guide, T4 datasheet, Nsight Compute Profiling Guide.
- EEP 524 Lectures 3–9, HW1–HW4, the final project specification and starter kit.
