# #95 Fused Quant+GEMM with Split-K and remap_xcd

## Hypothesis
Custom fused kernel that combines:
1. In-register bf16→FP4 quantization (saves quant kernel launch)
2. remap_xcd() for XCD-aware scheduling (missing from gemm_a16wfp4)
3. Split-K support via temp fp32 buffer + separate reduce kernel

Target: M=16 K=7168 (21.1µs) where current path is quant + gemm_afp4wfp4.
Saves 1 kernel launch (~1-2µs) while keeping remap_xcd benefits.

## Architecture
- KSPLIT=1: Same as previous fused kernel but with remap_xcd
- KSPLIT>1: Main kernel writes fp32 partials to (KSPLIT,M,N) buffer,
  reduce kernel sums to bf16 output
- M>16: Falls through to standard quant + gemm_afp4wfp4 path

## Results (Ranked)
- M=4: 14.5µs (**WORSE** +2.1µs - extra JIT overhead from 3 kernel definitions)
- M=16: 19.6µs (**BETTER** -1.5µs! Fused quant+remap_xcd works!)
- M=32, N=4096: 12.7µs (same)
- M=32, N=2880: 12.6µs (same)
- M=64: 20.6µs (same)
- M=256: 20.6µs (same)

KEY: M=16 K=7168 improves 1.5µs by fusing quant + using remap_xcd.
But M=4 regresses 2.1µs from extra @triton.jit compilation overhead.
Net geomean: ~16.0µs (approximately same as #92's 16.1µs)
