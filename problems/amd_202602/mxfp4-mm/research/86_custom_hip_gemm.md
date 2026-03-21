# #86 Custom HIP FP4 GEMM via load_inline

## Breakthrough
- load_inline compiles on gfx950 with PYTORCH_ROCM_ARCH=gfx950 (#85 confirmed)
- All top AMD fp8-gemm submissions use load_inline + custom HIP kernels
- AMD CDNA4 blog provides complete optimization trajectory

## Architecture (from AMD blog)
1. MFMA instruction: __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4
   - FP4 type = 4 (e2m1)
   - 32x32 output tile, 64 K elements per instruction
   - E8M0 scaling built into the instruction
2. Thread block: 256 threads (4 waves of 64)
3. Output tile: start with 32x128 (M=32, N=128)
4. K tile: 128 elements (2 MFMA instructions per K step)
5. LDS: load A and B tiles, apply swizzle
6. Double buffer K dimension

## Plan
1. Start with matrix-core baseline (no double buffering)
2. Get correctness working first
3. Then optimize with vectorized loads, double buffering
4. Finally 8-wave ping-pong if needed

## Key FP4 MFMA intrinsic
```cpp
// V_MFMA_SCALE_F32_32X32X64_F8F6F4
// d = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
//     a, b, c,         // a,b are packed fp4 data, c is f32 accumulator
//     4, 4,            // Atype=4(fp4), Btype=4(fp4)
//     0, scale_a,      // OPSEL_A=0, scale_a is e8m0
//     0, scale_b       // OPSEL_B=0, scale_b is e8m0
// )
```
