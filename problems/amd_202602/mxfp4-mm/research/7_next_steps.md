# Attempt 7: Next Optimization Directions

## Current Status
- Hybrid approach: ~20.1µs geomean (17% improvement)
- K≤512: 35% faster with Triton
- K>512: No improvement (using CK with default config)

## Bottleneck Analysis
The large K cases (K=1536, 2048, 7168) are unchanged because:
1. CK kernel has no tuned config for our shapes
2. Triton requires double quantization (for unshuffled B_scale)

## New Ideas to Explore

### 1. Reverse e8m0_shuffle
If we can "unshuffle" B_scale_sh back to B_scale, we can:
- Use Triton for ALL cases
- Avoid double quantization overhead
- Potentially get tuned Triton performance for large K

### 2. Direct ASM kernel with split-K
The logs show `gemm_a4w4_asm` has `log2_k_split` parameter.
Could improve large K performance by parallelizing across K dimension.

### 3. Custom Triton config
Pass explicit config dict to `gemm_afp4wfp4` with optimized:
- BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
- NUM_KSPLIT for split-K parallelism

### 4. Lower K threshold
Try K≤1024 to use Triton for K=1536 case.

## Result
Pending...
