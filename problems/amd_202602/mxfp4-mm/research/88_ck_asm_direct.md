# #88 CK ASM Direct

## Hypothesis
Use CK ASM kernel (`aiter.gemm_a4w4`) for M>=32 shapes. The reference kernel uses this path.
We already have B_shuffle and B_scale_sh in the right format, so no extra preprocessing.
For A, we need e8m0_shuffle which adds ~1µs overhead.

CK ASM has tuned configs for:
- M=64, N=7168, K=2048 (32x128 tile, splitK=0)
- M=256, N=3072, K=1536 (32x128 tile, splitK=0)
- M=32 shapes fall back to padded M configs

## Changes
- M>=32: Use aiter.gemm_a4w4() with B_shuffle + B_scale_sh
- M<=16: Keep fused quant+GEMM Triton kernel
- Still inject Triton configs (for M<=16 shapes that use Triton)

## Expected
If CK ASM is better tuned, the large-K shapes (21µs) should improve.
Main risk: e8m0_shuffle overhead + potential worse default configs for shapes without tuning.

## Results
MUCH WORSE. CK ASM is ~2x slower than Triton for M>=32:
- M=32, N=4096, K=512: 24.8µs (was 12.6µs with Triton) ❌
- M=32, N=2880, K=512: 24.7µs (was 12.5µs) ❌
- M=64, N=7168, K=2048: 30.2µs (was 21.7µs) ❌
- M=256, N=3072, K=1536: 28.6µs (was 20.6µs) ❌
Ranked geomean: ~22.5µs (was ~16.2µs)

CONCLUSION: CK ASM with default configs is NOT competitive.
The Triton kernel with injected configs is much better.
