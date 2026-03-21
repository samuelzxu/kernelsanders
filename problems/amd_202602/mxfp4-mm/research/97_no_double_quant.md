# #97 No Double-Quant (e8m0_unshuffle for all K)

## Hypothesis
Use e8m0_unshuffle(B_scale_sh) for ALL K values instead of dynamic_mxfp4_quant(B)
for K<=512. Saves the quant kernel for K=512 shapes.

## Results (Ranked)
WORSE overall:
- M=4, K=512: 13.4µs (was 12.4µs) ❌ +1.0µs
- M=32, K=512: 14.0/13.9µs (was 12.7/12.6µs) ❌ +1.3µs
- M=16, K=7168: 20.9µs (was 21.1µs) ✓ -0.2µs
- M=64, K=2048: 20.4µs (same)
- M=256, K=1536: 20.4µs (same)

## Analysis
e8m0_unshuffle is SLOWER than dynamic_mxfp4_quant(B) for K=512:
- unshuffle: view→reshape→permute→contiguous (GPU copy)→reshape→slice
- The .contiguous() call allocates+copies, more Python overhead
- dynamic_mxfp4_quant: single fused GPU kernel, well-optimized

CONCLUSION: Keep the K<=512/K>512 split from #92. The double-quant is faster.
