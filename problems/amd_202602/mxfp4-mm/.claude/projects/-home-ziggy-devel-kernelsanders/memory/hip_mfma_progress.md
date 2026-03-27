---
name: hip_mfma_progress
description: HIP MFMA FP4 GEMM kernel debugging progress for mxfp4-mm competition
type: project
---

## HIP MFMA FP4 GEMM Kernel - What We Know (2026-03-21)

### CONFIRMED via probes (#240, #241):
- **A register layout**: lane l32 → row l32, group (lane/32) → 16-byte K-half. Confirmed correct.
- **B register layout**: IDENTICAL to A. lane l32 → col l32. No nibble repacking needed.
- **Output mapping**: acc[i*4+j] → row=(grp*4+i*8+j), col=l32. Confirmed correct.
- **Scale packing**: `(int)scale0 | ((int)scale1 << 8)` with opsel=0. Uniform scale test showed no effect of changing byte[1].

### CONFIRMED via diagnostics (#243, #244):
- **A_scale from dynamic_mxfp4_quant**: Has transposed strides (1, M). Must call .contiguous() to fix.
- **A_q strides**: Row-major (3584, 1). No issue.
- **B_scale unshuffle**: Perfect match with fresh dynamic_mxfp4_quant(B) output. 0/473088 mismatches.
- **B_q vs B_shuffle**: Same sorted bytes, same sorted nibbles. Same row0 data.

### REMAINING BUG:
- Errors are ~20-50% off reference, consistently across all test shapes
- Scale packing changes (uniform vs dual) have NO EFFECT on output
- Data formats are all verified correct
- The discrepancy likely comes from the quantization algorithm difference: dynamic_mxfp4_quant uses a C++ JIT kernel, while the preshuffle reference uses Triton's inline _mxfp4_quant_op
- OR: the MFMA's internal FP4 interpretation differs from what we assume (nibble order, denorm handling, etc.)

### Next steps:
1. Submit tiny GEMM element-by-element comparison probe
2. Try using the V_MFMA_F32_32X32X64_F8F6F4 (unscaled) variant with explicit scale multiplication
3. Try feeding pre-quantized + shuffled A/B data (matching the gemm_a4w4 reference path exactly)

**Why:** Competition leader is at 8.2µs geomean vs our 12µs. K=1536 M=256 at 19.3µs is the bottleneck. A working custom HIP MFMA kernel could potentially reach 10-12µs for this shape.

**How to apply:** When resuming HIP kernel work, start from experiment #242 and the probes #240/#241/#243/#244.

## Session 2 Updates (2026-03-21 continued)

### Gluon compilation attempts:
- Triton 3.6.0+rocm7.2 (rocm/pytorch:latest): `LLVM ERROR: Invalid basis 128` - layout format incompatible
- Triton 3.5.0 (PyPI, ghcr.io/gpu-mode/amd-runner:main): `invalid intrinsic shape` - lacks gfx950 MFMA support
- Need ROCm fork of Triton 3.5.x (release/internal/3.5.x branch) built from source - OOM on 31GB machine
- **Gluon path dead without more RAM or a machine with ROCm Triton pre-installed**

### HIP kernel B_shuffle test (#246):
- Loading from preshuffle format `(N//16, K*8)` with stride indexing gave COMPLETELY wrong results
- The preshuffle packing is NOT simple N-interleave; it's tied to Triton's tile access pattern
- B_q direct loading (#242) remains the closest approach (~20-50% errors from quant difference)

### Root cause confirmed:
- All data formats verified correct (register layout, scales, B_q bytes, strides)
- The ~20-50% error is from `dynamic_mxfp4_quant` producing different A quantization than the preshuffle kernel's inline `_mxfp4_quant_op`
- The inline quant in the Triton kernel uses a different rounding/scaling algorithm
- To fix: need to replicate the exact inline quant in HIP, OR find the `dynamic_mxfp4_quant` C++ source (it's JIT-compiled, not in Python)

## Session 3: Hardware Quant Intrinsic (2026-03-21)

### Major breakthrough: `__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16`
- Found in CK source: `aiter/jit/build/ck/include/ck_tile/core/numeric/pk_fp4.hpp`
- Hardware instruction for bf16→FP4 conversion with scale
- Usage: `__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(u32, bf16x2_pair, float_scale, const_word_idx)`
- bf16x2 must use `__bf16` type (not `hip_bfloat16`), `word_idx` must be compile-time constant

### Results with hw quant (#247c/d):
- Errors reduced from ~40% (software quant) to ~15% (hw quant)!
- Some elements now match perfectly
- Nibble pair order confirmed: `{v0, v1}` is correct (swapping makes it much worse)
- E8M0 scale computation change (ceil vs biased exponent) had NO effect

### Remaining ~15% error:
- Scale computation mismatch between our mk_e8m0() and the reference's formula
- Some groups match, others are off - suggests off-by-one in exponent for certain ranges
- Need the exact `_mxfp4_quant_op` source (not found in runner's aiter - it may be defined in the eval.py/task.py upload process)
- The reference formula might be: e8m0 = floor(log2(max_abs)) + 3 + 127 (since max FP4 E2M1 = 6 = 1.5 * 2^2)

### E8M0 scale formula iterations (hw quant intrinsic):
| Version | Formula | K=512 (0,0) | K=512 (0,1) | Notes |
|---------|---------|-------------|-------------|-------|
| 247c | `ceil(log2(max/6))+127` | -16.5 vs -14.25 (16%) | passes | **BEST overall** |
| 247d | biased_exp-2+adjust | identical to 247c | same | Scale change no effect |
| 247f | raw biased_exp (floor(log2)+127) | -14.875 vs -14.25 (4%) | 38.25 vs 41.0 | Closest for some, worse for others |
| 247g | ceil(log2(max))+127 | -18.5 vs -14.25 | passes | Worse |
| 247h | INVERSE scale (1/scale) | -2.4375 vs -14.25 (6x too small) | | Confirms scale direction correct |
| 247e | nibble swap {v1,v0} | -93.0 vs 163.0 | | Much worse, wrong order |

### Current best file: 247_hip_hwquant.py
- Module name: mfma247h (needs increment for next test)
- Scale formula: reverted to 247c (ceil(log2(max/6))+127) with scale = e8m0_to_f(e8m0)
- HW quant: `__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16` with `{v0, v1}` order
- Remaining ~15% errors: caused by aiter Issue #974 (incorrect MXFP4 mantissa rounding)
- The hw intrinsic has CORRECT rounding; the mismatch is between hw-quant A and software-quant B (from dynamic_mxfp4_quant)

## NEW LEAD: hipBLASLt FP4 (2026-03-22)
- ROCm 7.0+ has HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 for MXFP4
- RadeonFlow winner used hipBLASLt algorithm enumeration (github.com/RadeonFlow/RadeonFlow_Kernels)
- Runner confirmed: Triton 3.6.0, hip/gfx950, triton_kernels NOT installable
- hipBLASLt probe #255 pending (rate limited)
