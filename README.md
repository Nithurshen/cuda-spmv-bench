# cuda-spmv-bench

CSR sparse matrix-vector multiply (SpMV) on an NVIDIA GPU: thread-per-row, warp-per-row and sub-warp kernels, checked against a CPU reference, timed with CUDA events, compared to cuSPARSE, and profiled with Nsight Compute.

I built this to practise profiler-driven kernel work on an irregular, memory-bound problem. The short version: **the best thread-to-row mapping depends on row length, and the profiler shows why.**

## Summary

- On a matrix with long rows, warp-per-row is about **3.3x faster** than thread-per-row and matches cuSPARSE.
- On a matrix with very short rows, the same kernel is about **2x slower** than thread-per-row, because most of its 32 lanes sit idle.
- A **sub-warp kernel** (VEC lanes per row) fixes this. Picking VEC as the smallest power of two at least the mean row length chose VEC = 4 on the short-row matrix and reached cuSPARSE's speed.

## Setup

RTX A4000 (sm_86), CUDA 12.8, Windows, fp32 with 32-bit indices. Matrices come from the SuiteSparse collection in Matrix Market format. The input vector is random in [-1, 1] with a fixed seed.

| Matrix | Shape | Nonzeros | Mean nnz/row | Rows |
|---|---|---|---|---|
| `cant` | about 62K x 62K | about 4.0M | about 64 | long, regular |
| `webbase-1M` | about 1.0M x 1.0M | about 3.1M | about 3 | very short, uneven |

## Kernels

- **csr-scalar**: one thread per row. Neighbouring threads read different rows, so loads are not coalesced.
- **csr-vector**: one warp per row. Lanes read consecutive nonzeros, then a `__shfl_down_sync` tree reduces the sums.
- **subwarp-VEC**: the same with VEC = 1 to 32 lanes per row, so several rows share a warp.
- **cuSPARSE**: `cusparseSpMV`, default algorithm, as the baseline. I time the whole call, which launches several kernels.

Every kernel is checked against a double-precision CPU result (relative error under 1e-3) before timing.

## Method

- Timing: CUDA events, 5 warm-up runs and 50 timed runs. I report the minimum.
- Bandwidth: `bytes / min_time`, with bytes = `8*nnz + 4*(rows+1) + 4*cols + 4*rows`. This is the algorithm's byte count, not measured DRAM traffic.
- Profiling: Nsight Compute. I read DRAM throughput as a percentage of peak, and global load sectors divided by requests as a coalescing signal.

## Results

| Matrix | Kernel | GB/s | DRAM % of peak | Sectors per request |
|---|---|---|---|---|
| `cant` | csr-scalar | 110 | 47 | 19.7 |
| `cant` | csr-vector | 364 | 77 | 4.0 |
| `cant` | cuSPARSE | 356 | n/a | n/a |
| `webbase-1M` | csr-scalar | 104 | 21 | 4.0 |
| `webbase-1M` | csr-vector | 56 | 9.5 | 1.4 |
| `webbase-1M` | subwarp-4 (auto) | 295 | 60 | 2.7 |
| `webbase-1M` | cuSPARSE | 297 | n/a | n/a |

Sub-warp sweep on `webbase-1M` (GB/s): VEC 1: 104, 2: 169, 4: 295, 8: 212, 16: 117, 32: 60. On `cant`, VEC 8 and 16 reach 377 GB/s, a few percent above VEC 32.

## Analysis

- **Long rows (`cant`).** The scalar kernel needs about 20 sectors per load request, the signature of uncoalesced access, and reaches only 47% of peak DRAM throughput. The warp-per-row kernel needs 4, reaches 77%, and is 3.3x faster.
- **Short rows (`webbase-1M`).** With about 3 nonzeros per row, a 32-lane warp leaves most lanes idle. It issues 5.1M load requests against 1.1M for the scalar kernel, and DRAM utilization falls to 9.5%.
- **The sweet spot follows row length.** Throughput peaks where VEC is near the mean row length: 4 on `webbase-1M`, 8 to 16 on `cant`. I did not isolate why VEC 8 to 16 beats 32 on `cant`, and the gap is small enough to be partly noise.