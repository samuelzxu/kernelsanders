# v82: cktile sk=1 for E=257 bs=512 d=256

## Key Insight
E=257 bs=512 has 17 tok/exp with inter_dim=256.
Each expert processes ~18 tokens with K=7168, N=512.
This is a skinny GEMM (small M) that's likely memory-bound.
For memory-bound GEMMs, FP4 vs BF16 compute throughput doesn't matter.
Eliminating quant (~20µs) may save more than the FP4 advantage.

## Difference from v74 (which was 7% slower)
v74 used sk=2 (split_k overhead). v82 uses sk=1 (no split).
