# Mixed-MLA Decode Kernel Optimization — Continuation Prompt

## The Problem
Optimize an MLA (Multi-head Latent Attention) decode kernel for AMD MI355X (gfx950) GPU for the GPU MODE competition (`amd-mixed-mla` leaderboard). Minimize the geometric mean of 8 benchmark configurations: bs={4,32,64,256} × kv={1024,8192}.

**Current leaderboard score: ~70µs geomean. Top competitor: ~13.5µs.**

## Submission Command
```bash
popcorn-cli submit --leaderboard amd-mixed-mla --mode test --gpu MI355X submission.py --no-tui
popcorn-cli submit --leaderboard amd-mixed-mla --mode benchmark --gpu MI355X submission.py --no-tui
popcorn-cli submit --leaderboard amd-mixed-mla --mode leaderboard --gpu MI355X submission.py --no-tui
```
Rate limits: 1/hour leaderboard, 6/hour benchmark/test.

**CRITICAL RULES:**
- The word "stream" MUST NOT appear anywhere in submission.py (instant rejection)
- No cross-invocation caching of metadata/tensors (deemed illegitimate)
- Pre-allocated constant tensors at module level (like `torch.arange`) ARE allowed
- Pre-compiled kernels that perform genuine computation ARE allowed
- Do not resubmit our best version unnecessarily — leaderboard auto-selects best

## MLA Architecture
- NH=16 query heads, NKV=1 KV head, DQ=576 (512 nope + 64 rope), DV=512
- SM_SCALE = 1/sqrt(576)
- Correctness tolerance: rtol=2e-02, atol=8e-03
- KV data available in 3 formats: bf16, fp8 (per-tensor scale), mxfp4 (block-32 E8M0 scales)

## Current Best Submission: `research/177_kvi_after_compile.py` (~70µs)
Architecture:
- **Small configs** (bs≤4, or bs≤32 & kv≤1024): `torch.compile` GEMM attention (baddbmm + softmax + bmm)
- **Large configs**: fp8 assembly via `aiter.mla_decode_stage1_asm_fwd` + `aiter.mla_reduce_v1`
  - kv≤1024: bf16 Q (a16w8 kernel), kvg=32, nks=8
  - kv>1024: fp8 Q via `dynamic_per_tensor_quant_fp8_i8`, kvg=32, nks=32
- Key optimization: `_KVI_SHORT = torch.arange(256*1024, device="cuda")` pre-allocated at module level, used as `_KVI_SHORT[:n]` (zero-cost view) for kv≤1024 configs, saving ~4µs vs `torch.arange(n)` per call

## Benchmark Numbers (attempt 177, ranked)
| Config | Time (µs) |
|--------|-----------|
| bs=4/kv=1024 | 19.4 |
| bs=4/kv=8192 | 39.3 |
| bs=32/kv=1024 | 33.9 |
| bs=32/kv=8192 | 90.9 |
| bs=64/kv=1024 | 46.5 |
| bs=64/kv=8192 | 144 |
| bs=256/kv=1024 | 109 |
| bs=256/kv=8192 | 339 |
| **Geomean** | **~70µs** |

## What We Tried and DEFINITIVELY Ruled Out

### 1. MXFP4 Q@K^T via `tl.dot_scaled` — PASSES test mode but FAILS some benchmark seeds
- The Triton kernel compiles and works: split 576-dim into nope(512)+rope(64), quantize Q via `_mxfp4_quant_op`, load K as (K_packed, N) for RHS
- **BUT: MXFP4 precision (4-bit) is too lossy for Q@K^T.** Fails seeds 5415 (bs=32/kv=8192) and 4220 (bs=4/kv=8192) with 459-594 mismatched elements. The FP4 quantization noise on attention scores gets amplified by softmax.
- **This rules out the entire MXFP4 path for Q@K^T** — not just V but K too.
- Research files: `256_transposed_k.py` (first working version), `260_mxfp4_v_dequant.py`, `261_dot_scaled_pv.py`

### 2. MXFP4 V dequantization — FAILS precision
- Both HIP kernel dequant AND pure PyTorch dequant produce identical errors
- 594 mismatched elements for seed 5415, same elements, same magnitudes
- NOT a kernel bug — fundamental precision limitation of FP4 V quantization
- Research: `260_mxfp4_v_dequant.py` (Triton software dequant, 1625µs — too slow AND too imprecise)

### 3. fp8 V with MXFP4 K — ALSO FAILS
- Even fp8 V (8-bit, per-tensor scale) fails when combined with MXFP4 Q@K^T
- Root cause: the MXFP4 Q@K^T scores are the problem, not V precision
- Research: submission with `_run_mxfp4_fp8v` function

### 4. `tl.dot_scaled` with `rhs_k_pack=False` — NOT IMPLEMENTED
- `LLVM ERROR: Unsupported DotScaleOp found when converting TritonGPU to LLVM`
- AMD Triton backend does not implement `rhs_k_pack=False`
- This was the last hope for hardware MXFP4 V dequant via Triton

### 5. Software FP4 V dequant in Triton — TOO SLOW
- Bit manipulation (& 0xF, >> 4, exp2, where) per nibble adds massive compute overhead
- Two separate tl.dot calls (even/odd DV) double compute and register pressure
- bs=256/kv=8192: 1625µs (MXFP4 V dequant) vs 899µs (bf16 V) vs 332µs (assembly)

### 6. bf16 Q for kv>1024 assembly (a16w8 vs a8w8) — SLOWER
- bs=256/kv=8192: 617µs (a16w8) vs 332µs (a8w8). The a8w8 fp8 MFMA is much faster.

### 7. nks/kvg tuning — MARGINAL
- Tested nks=16 for bs>=128: 343µs vs 341µs for bs=256/kv=8192 — negligible
- Tested kvg=16/nks=16 for bs<=32: slightly worse
- The assembly kernel is bandwidth-bound; tuning dispatch params doesn't help

### 8. Non-persistent assembly mode — FAILS for bf16 Q + fp8 KV
- `RuntimeError: cannot get heuristic kernel! q_type:bf16 kv_type:fp8`
- No a16w8 kernel available in non-persistent mode

### 9. Metadata caching — BANNED by competition rules
- Caching metadata tensors across invocations = cross-invocation state = illegitimate
- The cached versions achieved ~40µs (vs 70µs uncached) — 30µs of pure overhead per call

## What IS Working / Available

### Docker Cross-Compilation Pipeline (VERIFIED)
```bash
docker run --rm --platform linux/amd64 \
  -v $(pwd):/workspace rocm/pytorch:latest \
  bash -c "hipcc -O3 --offload-arch=gfx950 -shared -fPIC kernel.hip -o kernel.so"
```
- `rocm/pytorch:latest` has ROCm 7.2 with gfx950 support (45.9GB image, already pulled)
- **DO NOT use** `ghcr.io/gpu-mode/amd-runner:main` — has ROCm 6.3, no gfx950 support
- On a Linux machine with AMD GPU, you can compile directly with hipcc

### GitHub Repo for Precompiled Kernels
**Repo:** https://github.com/samuelzxu/gfx950-kernels (release v0.1)
- Download URL pattern: `https://github.com/samuelzxu/gfx950-kernels/releases/download/v0.1/{filename}`
- Runtime download verified on runner via `urllib.request.urlretrieve`
- ctypes loading verified: `lib = ctypes.CDLL("/tmp/kernel.so")`
- HIP device kernels need host wrapper functions (extern "C" with `<<<>>>` launch syntax)

### Compiled HIP Kernels (in `research/`)
- `v_dequant_v3_gfx950.so` — MXFP4 V→bf16 dequant with stride support (has correctness bug from float→bf16 truncation vs round-to-nearest-even, AND MXFP4 precision is too low anyway)
- `mxfp4_qk_gfx950.so` — MFMA FP4 Q@K^T score computation (skeleton)
- `mxfp4_attention_gfx950.so` — MFMA FP4 test kernel
- `test_gfx950.so` — simple vector add (pipeline validation)

### MFMA FP4 Intrinsic Reference (gfx950 CDNA4)
```cpp
// 32x32 output tile, 64 FP4 elements along K per instruction
__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
    a_i32x8, b_i32x8, acc_f32x16,
    4, 4,       // fmt_a=FP4(E2M1), fmt_b=FP4(E2M1)
    0, scale_a, // opsel=0, E8M0 scale
    0, scale_b  // opsel=0, E8M0 scale
);
// Format codes: 0=E4M3, 1=E5M2, 2=E2M3, 3=E3M2, 4=E2M1(FP4)
// Also available: 16x16x128 variant with f32x4_t accumulator
```

### Key Reference Files in aiter
- `aiter/ops/triton/_triton_kernels/attention/mla_decode_rope.py` — Triton MLA decode kernel
- `aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention_mxfp4.py` — MXFP4 attention (uses bf16 V, not MXFP4 V)
- `aiter/ops/triton/_triton_kernels/quant/quant.py` — `_mxfp4_quant_op`
- `aiter/ops/triton/_triton_kernels/fusions/fused_bmm_rope_kv_cache.py` — Working `tl.dot_scaled` usage
- `aiter/csrc/include/opus/opus.hpp` (lines 1770-1897) — Opus MFMA wrapper
- `aiter/op_tests/opus/device/test_mxfp.cu` — FP4 MFMA test with register layout details
- `aiter/hsa/gfx950/mla/mla_asm.csv` — Assembly kernel configurations (a8w8, a16w8, a16w16 only — NO MXFP4)
- `aiter/csrc/cpp_itfs/hsaco_launcher.py` — Python hip-python based HSACO launcher
- `aiter/utility/fp4_utils.py` — Reference `mxfp4_to_f32()` and `e8m0_to_f32()` implementations

### tl.dot_scaled Lessons (for future Triton kernel work)
- **RHS must be loaded as (K_packed, N)** not (N, K_packed) — transpose the pointer pattern
- **RHS scales stay as (N, K//32)** — NOT transposed
- **PADDED_H=32** for MFMA 32x32 tile requirement (actual NH=16)
- **Split 576-dim** into nope(512) + rope(64) — both power-of-2
- **Q quantized inside kernel** via `_mxfp4_quant_op` — produces MFMA-compatible scale layout
- **Triton doesn't support `continue`** — use `if condition:` block instead
- **`rhs_k_pack=False` NOT implemented** on AMD Triton LLVM backend

## Overhead Analysis (where the 70µs comes from)

For bs=256/kv=8192 (339µs):
- Theoretical bandwidth minimum: 256×8192×576 bytes / 8TB/s ≈ 150µs
- Actual: 339µs (2.26× theoretical)
- Overhead breakdown estimate: metadata ~5µs, tensor alloc ~5µs, Q quant ~3µs, stage2 reduce ~30µs, kernel launch/scheduling ~10µs, memory access inefficiency ~136µs

For bs=4/kv=1024 (19.4µs):
- Data: 4×1024×576 = 2.4MB → bandwidth time ≈ 0.3µs
- Nearly all 19µs is overhead (kernel launch, torch.compile dispatch, etc.)

## Paths Forward (Prioritized)

### 1. Custom HIP Flash-Decode Kernel (HIGH IMPACT, HIGH EFFORT)
Write a single fused HIP C++ kernel that:
- Reads fp8 KV data (576 bytes/token) — same format as assembly
- Uses fp8 MFMA for Q@K^T (NOT FP4 — precision requirement)
- Does online softmax + V accumulation in single pass
- Eliminates: Python overhead, metadata computation, tensor allocation, separate reduce kernel
- Potential: each call is just `hipLaunchKernel` → could save 30-50µs overhead
- Challenge: MFMA register layout, online softmax in HIP, split-K reduce
- With Linux + AMD GPU, you can test locally (not just cross-compile)

### 2. Reduce Per-Call Overhead (MEDIUM IMPACT, MEDIUM EFFORT)
The ~30µs overhead per assembly call comes from:
- `get_mla_metadata_info_v1` + `get_mla_metadata_v1` — GPU kernel launches for metadata
- `torch.empty(...)` × 6 tensors — allocator overhead
- `aiter.dynamic_per_tensor_quant` — Q quantization kernel
- `aiter.mla_reduce_v1` — separate reduce kernel
Ideas:
- Pre-allocate ALL tensors at module level (like `_KVI_SHORT` but for lg, ls, o, metadata)
- This is NOT metadata caching — it's buffer reuse. The tensors are overwritten each call.
- But be careful: if shapes change across configs, need per-config buffers

### 3. Use qh128 Assembly Variant (SPECULATIVE)
The `mla_a8w8_qh128_*` assembly kernels process 128 query heads per block. Could batch 8 batch elements (8×16=128 heads) into one block, reducing kernel launches by 8×. Requires careful metadata setup.

### 4. Triton MLA Decode Kernel (ALTERNATIVE)
Adapt `aiter/ops/triton/_triton_kernels/attention/mla_decode_rope.py` directly. This is aiter's own optimized Triton MLA decode kernel. It already handles the full MLA decode with split-K and online softmax. Could be configured for fp8 KV with per-tensor scale.

### 5. Precompile AITER JIT Modules (RELIABILITY)
The runner spends 220s on JIT builds (module_mla_metadata: 31s, module_mla_asm: 20s, module_mla_reduce: 170s, module_quant: 24s). Pre-building and hosting these .so files would eliminate cold start risk. Couldn't build them in Docker without GPU — on a Linux machine with AMD GPU, run the build script and upload to GitHub release.

## File Structure
```
mixed-mla/
├── submission.py              ← Current best (attempt 177)
├── task.py                    ← Problem definition (DO NOT MODIFY)
├── reference.py               ← Reference implementation
├── PROMPT.md                  ← THIS FILE
├── research/
│   ├── 177_kvi_after_compile.py  ← Best legitimate submission
│   ├── 257_hybrid_compile_assembly.py  ← Clean hybrid (no _KVI_SHORT)
│   ├── 256_transposed_k.py    ← First working MXFP4 Triton kernel
│   ├── 260_mxfp4_v_dequant.py ← Software V dequant (too slow)
│   ├── 261_dot_scaled_pv.py   ← rhs_k_pack=False attempt (LLVM error)
│   ├── 262_hip_download_test.py ← Network download validation
│   ├── v_dequant_v3.hip       ← HIP V dequant source
│   ├── v_dequant_v3_gfx950.so ← Compiled V dequant (has precision issues)
│   ├── mxfp4_qk_kernel.hip    ← MFMA FP4 Q@K^T source
│   ├── mxfp4_qk_gfx950.so     ← Compiled Q@K^T kernel
│   ├── mxfp4_flash_decode.hip  ← Flash-decode skeleton
│   ├── mxfp4_attention.hip     ← MFMA FP4 test kernel
│   ├── SUMMARY.md              ← Historical optimization summary
│   └── *.md                    ← Notes for each attempt (1-262+)
```

## Key Learnings
1. **MXFP4 (FP4) precision is insufficient** for this competition's tolerance. Both Q@K^T scores and V values have too much quantization noise. The reference uses bf16/fp8.
2. **The assembly kernel is bandwidth-bound** at ~2.26× theoretical bandwidth. Tuning dispatch parameters (nks, kvg) gives <2% improvement.
3. **Per-call overhead is ~30µs** (metadata + alloc + quant + reduce). This is 43% of the small-config time and limits the geomean.
4. **Docker cross-compilation works** for gfx950 HIP kernels. The full pipeline (compile → GitHub → download → ctypes) is validated.
5. **The gap to #1 is 5×** (70µs vs 13.5µs). This requires a fundamentally different approach, likely a custom fused kernel that eliminates all Python/aiter overhead.
