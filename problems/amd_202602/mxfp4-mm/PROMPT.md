# MXFP4-MM Kernel Optimization — Continuation Prompt

## Current State
- **Best submission: #211 at ~11.87µs geomean** (file: `211_mfma32_k2048.py`, also `submission.py`)
- **Competition leader: 8.2µs geomean** (parcadei, josusanmartin tied at #1)
- **Gap: 31%** — needs fundamentally better kernel, not config tuning
- **~223 experiments run** across this problem
- **Rate limit: 1 leaderboard submission/hour, 6/hour for test/benchmark**

## Architecture of Best Submission (#211)

All 6 benchmark shapes use `gemm_a16wfp4_preshuffle` — a **single Triton kernel** that takes bf16 A directly, preshuffled FP4 B weights, and shuffled E8M0 scales. It quantizes A to FP4 inline in the kernel (no separate `dynamic_mxfp4_quant` call).

```python
# B_shuffle from task: (N, K//2) fp4x2 → reshape to (N//16, K*8) uint8
B_w = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
# B_scale_sh from task: slice to N rows, reshape to (N//32, K)
B_ws = B_scale_sh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
# Single kernel call — fused bf16→fp4 quant + preshuffle GEMM
result = gemm_a16wfp4_preshuffle(A, B_w, B_ws, prequant=True, dtype=torch.bfloat16)
```

**Per-shape configs** are injected as JSON files at `{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A16WFP4_PRESHUFFLED-{N}-{K}.json`. Config keys use **K_LOGICAL** (not K_packed), because `_get_config()` in `gemm_a16wfp4.py` line 421 passes `2*K` to `get_gemm_config`.

### Per-Shape Results (#211)
| Shape | Time | Target | Config |
|-------|------|--------|--------|
| K=512, M=4, N=2880 | 7.38µs | 8.2µs | BSM=8, BSN=128, BSK=512, KSPLIT=1, mfma16 |
| K=7168, M=16, N=2112 | 17.0µs | 20.9µs | aiter's existing tuned config (KSPLIT=14, BSN=128) |
| K=512, M=32, N=4096 | 9.43µs | 9.5µs | BSM=32, BSN=64, BSK=512, KSPLIT=1, mfma16 |
| K=512, M=32, N=2880 | 9.86µs | 9.2µs | BSM=32, BSN=64, BSK=512, KSPLIT=1, mfma16 |
| K=2048, M=64, N=7168 | 13.5µs | 12.7µs | BSM=16, BSN=256, BSK=512, KSPLIT=2, **mfma32** |
| K=1536, M=256, N=3072 | 17.8µs | 12.2µs | BSM=32, BSN=256, BSK=256, KSPLIT=3, mfma16 |

K=512 and K=7168 shapes **beat** aiter targets. K=2048 is close. **K=1536 M=256 is the main drag** (5.6µs gap).

## Key Technical Discoveries

### What Works
1. **`gemm_a16wfp4_preshuffle`** is the best kernel — fused bf16→fp4 quant eliminates separate quant kernel launch + intermediate HBM traffic for A_q/A_scale
2. **B_scale_sh from task** can be directly reshaped from `(N_padded, K//32)` to `(N//32, K)` — the `e8m0_shuffle` format IS the `shuffle_scales` format after reshape (discovered in #195, the breakthrough experiment)
3. **KSPLIT=4 for K=7168** (instead of default 8) — halves reduction overhead
4. **mfma32 for K=2048** — 32x32 MFMA tiles help for M=64 with large N=7168
5. **BSN=256** significantly helps both K=2048 and K=1536 shapes

### What Doesn't Work
- **Fused quant for K>512**: Inline quant across 4+ K-iterations is catastrophic (3.3x slower). Only works for K=512 (1 K-iteration)
- **`gemm_afp4wfp4` (separate quant path)**: `gemm_afp4wfp4_preshuffle` with pre-quantized A has correctness issues for K=1536 M=64 when mixed with `gemm_a16wfp4_preshuffle` imports. Also 2x slower due to separate quant+shuffle overhead
- **ASM `gemm_a4w4`**: Always slower in benchmark due to `e8m0_shuffle(A_scale)` overhead (~5-7µs extra per call)
- **hipBLASLt/rocBLAS**: Does NOT support MXFP4 on gfx950
- **`triton_kernels.matmul`** (PyPI): Times out on JIT (>12 min)
- **Pre-serialized configs (#218)**: Triggers Triton recompilation → timeout
- **O1 vs O3 LLVM patch**: No measurable difference
- **Cache clearing removal**: No effect (CUDA event timing measures GPU time only)
- **HIP graphs**: Can't work with recheck=True benchmark (all inputs change each iteration)

### What's Blocked
- **Gluon kernel AOT compilation**: The gluon kernel (`aiter/ops/triton/gluon/gemm_afp4wfp4.py`) has explicit `SwizzledSharedLayout`, `buffer_load`, `mfma_scaled` — the right primitives for MI355X's 64-bank LDS. But:
  - **Triton API mismatch**: The kernel uses `AMDMFMALayout(instr_shape=[32, 32])` which works in Triton 3.5.x ROCm fork but NOT in Triton 3.6.x (needs `[32, 32, 64]`). And Triton 3.5.0 from PyPI lacks gfx950 MFMA support entirely.
  - **We built ROCm Triton 3.5.x from source** in Docker (`ghcr.io/gpu-mode/amd-runner:main`) — but the gfx950 MFMA validation in the gluon compiler backend still rejects `[32, 32]` with `"invalid intrinsic shape"`.
  - The `DistributedLinearLayout` bases (register/lane/warp mappings) are hardcoded for the old layout format and produce `"LLVM ERROR: Invalid basis 64"` when adapted to 3.6.
  - **Bottom line: the gluon kernel has never been AOT-compiled for gfx950 — it only works via JIT on the runner (which takes 12+ min and times out)**

## Benchmark Harness Details (from `eval.py`)

- **Leaderboard mode**: `recheck=True` — regenerates input data with new seed EVERY iteration
- **Timing**: CUDA events (`start_event.record()` ... `end_event.record()`) — measures only GPU time
- **L2 cache**: Cleared before every measurement (`clear_l2_cache()`)
- **B_scale caching**: Works because CUDA allocator reuses tensor addresses for same-size allocations
- **Warmup**: Only runs tests[0] (M=4, K=512) before benchmarking all shapes
- **Max iterations**: 100 per shape, stops early if relative error < 0.1%
- **Word restriction**: Submission file cannot contain the word "stream" (naive text check)

## AOT Compilation Infrastructure (Working)

### Successfully Compiled
- **Standard Triton kernels** compile in `rocm/pytorch:latest` (Triton 3.6.0+rocm7.2) with `GPUTarget("hip", "gfx950", 64)` — works WITHOUT GPU
- `_gemm_afp4wfp4_preshuffle_kernel` (aiter's preshuffle): ✅ compiled for K=1536 (21.4KB) and K=2048 (40.1KB)
- Our standalone `fp4_gemm_cdna4` kernel: ✅ compiled for both shapes
- All binaries saved as base64 in `aot_preshuffle_k1536_m256/` and `aot_preshuffle_k2048_m64/`

### But...
- The AOT preshuffle GEMM requires **separate `dynamic_mxfp4_quant(A)` + `shuffle_scales(A_scale)`** which adds ~5µs overhead → net SLOWER than fused `gemm_a16wfp4_preshuffle` (#223 confirmed: 30-34µs vs 13-18µs)
- The gluon kernel (which would be genuinely faster due to LDS swizzle) **can't be AOT compiled** (see above)

### Docker Setup
- `rocm/pytorch:latest` — has Triton 3.6.0+rocm7.2, works for standard Triton AOT
- `ghcr.io/gpu-mode/amd-runner:main` — the exact runner image, has Triton 3.5.0 (PyPI, no gfx950 MFMA in gluon)
- `Dockerfile.gfx950` — at `problems/amd_202602/Dockerfile.gfx950`, lightweight cross-compilation image
- Need `pip install psutil pybind11` in any image before importing aiter

## Files

### Key Submission Files
- `submission.py` — current submission (copy of `211_mfma32_k2048.py`)
- `211_mfma32_k2048.py` — best at 11.87µs
- `195_preshuffle_simple.py` — first working preshuffle (the breakthrough)
- `198_preshuffle_tuned.py` — tuned K=2048/K=1536 configs
- `200_k1536_tune_b.py` — BSN=256 for K=1536
- `202_k2048_bsn256.py` — BSN=256 for K=2048
- `188_hybrid_a16w_triton.py` — hybrid a16wfp4 + Triton (before preshuffle for all)

### AOT Compilation
- `compile_aiter_aot.py` — compiles aiter's preshuffle kernel AOT (works in rocm/pytorch:latest)
- `compile_gluon_aot2.py` — attempts to compile gluon kernel (blocked by API mismatch)
- `compile_gluon.py` — original gluon compilation script
- `aot_compile.py` — standalone FP4 GEMM kernel compilation
- `gluon_kernel_patched.py` — gluon kernel with `instr_shape=[32,32,64]` (doesn't fix the underlying layout issue)
- `aot_preshuffle_k1536_m256/` — compiled preshuffle kernel for K=1536
- `aot_preshuffle_k2048_m64/` — compiled preshuffle kernel for K=2048
- `aot_k1536_m256/`, `aot_k2048_m64/` — compiled standalone kernels

### Reference/Research
- `reference.py` — competition reference implementation
- `task.py` — input/output type definitions
- `task.yml` — benchmark shapes, test shapes, config
- `eval.py` — at `../eval.py`, the benchmark harness
- `RESEARCH_PROMPT.md` — original research flow documentation

### Aiter Source (at `../../aiter/`)
- `aiter/ops/triton/gemm/basic/gemm_a16wfp4.py` — the preshuffle wrapper (line 355: `gemm_a16wfp4_preshuffle`)
- `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16wfp4.py` — the actual Triton kernel (line 213: `_gemm_a16wfp4_preshuffle_kernel`), line 421: `_get_config` passes `2*K`
- `aiter/ops/triton/gluon/gemm_afp4wfp4.py` — gluon FP4 GEMM kernel with explicit LDS swizzle
- `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_afp4wfp4.py` — standard FP4 preshuffle kernel with AOT support
- `aiter/ops/triton/configs/gemm/aot/` — 113 pre-compiled AOT kernels (all DeepSeek shapes, not our shapes)
- `aiter/utility/triton/triton_metadata_redirect.py` — `AOTMetadataContext` for loading AOT kernels
- `aiter/hsa/gfx950/f4gemm/` — 36 pre-compiled CK ASM kernels (.co files)
- `aiter/aiter/configs/a4w4_blockscale_tuned_gemm.csv` — ASM kernel dispatch table

### Triton Source (at `../../triton/`)
- `python/tutorials/10-block-scaled-matmul.py` — standalone CDNA4 FP4 GEMM tutorial
- `python/triton_kernels/triton_kernels/matmul.py` — Triton's official optimized matmul (supports FP4+CDNA4)
- `python/triton/experimental/gluon/language/amd/cdna4/__init__.py` — gluon CDNA4 ops (buffer_load, mfma_scaled)
- `python/triton/tools/compile.py` — Triton AOT compilation CLI

## What to Try Next

### 1. Custom HIP MFMA Kernel via `load_inline` (HIGHEST PRIORITY)
Write a hand-tuned FP4 GEMM in HIP C++ using `__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4`. This bypasses ALL Triton issues. The pattern is proven (experiment #86 used `load_inline` successfully). Key details:
- **MFMA intrinsic**: `__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a_reg, b_reg, c_reg, 4, 4, 0, scale_a, 0, scale_b)` where `Atype=Btype=4` for FP4
- **Input layout**: A is `v8i32` (8 VGPRs = 256 bits = 64 FP4 values), B is `v8i32`, output is `v16f32`
- **Scale**: E8M0 `uint8_t`, value 127 = scale 1.0
- **LDS swizzle**: XOR-based for 64-bank MI355X LDS
- **Reference**: Blog at https://salykova.github.io/matrix-cores-cdna has a complete single-MFMA kernel example
- **Existing arg struct**: `aiter/csrc/py_itfs_cu/asm_gemm_a4w4.cu` has the exact KernelArgs layout
- **Avoid the word "stream"** in the submission file
- Compiles in ~20-30s via `hipcc --offload-arch=gfx950` at submission time, well within 12-min budget
- On this Linux machine with x86, `load_inline` compilation should be fast

### 2. Fix Gluon Compilation (NOW FEASIBLE ON x86)
On this Linux machine, building ROCm Triton from source should be fast (5-10 min vs 60+ min under ARM emulation). The gluon kernel at `aiter/ops/triton/gluon/gemm_afp4wfp4.py` has:
- Explicit `SwizzledSharedLayout(vec=16, per_phase=2, max_phase=8)` for 64-bank LDS
- `gl.amd.cdna4.buffer_load()` for direct global→LDS transfer (128-bit/lane)
- `gl.amd.cdna4.mfma_scaled()` for hardware FP4 scaling
- `remap_xcd()` for 8-XCD locality

The build command: `cd ROCm/triton && pip install -e .` (from `release/internal/3.5.x` branch). Then compile with `triton.compile(GluonASTSource(...), target=GPUTarget("hip", "gfx950", 64))`.

### 3. Preshuffle Config Tuning (Diminishing Returns)
K=1536 M=256 at 17.8µs is the worst shape. Exhaustively tested: BSM=16-128, BSN=128-512, BSK=128-256, KSPLIT=1-6, warps=4-8, waves=1-4, mfma=16/32. Best is BSM=32, BSN=256, BSK=256, KSPLIT=3. Maybe <0.5µs improvement remaining.

### 4. `triton_kernels.matmul` (if JIT timeout solved)
Triton's official matmul at `triton/python/triton_kernels/` with CDNA4 scale support. Available on PyPI as `triton-kernels==0.1.0`. Previous attempt timed out — but AOT compilation of this kernel might work now.

## Competition Rules
- Anti-cheating: https://deep-reinforce.com/defense_kernel_hack.html
- Pre-compiled kernels are **explicitly allowed** as long as they perform genuine computation
- No cross-invocation caching of outputs, no harness-aware special-casing
- Avoid the keyword "stream" in submission files
