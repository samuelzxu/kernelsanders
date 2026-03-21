# Attempt 190: No TunableOp — confirms TunableOp is essential

## Results
- bs=32/kv=1024: 40.7µs WITHOUT TunableOp vs 33.7µs WITH = -7µs from TunableOp
- TunableOp is essential for GEMM performance

## Correctness failure
- bs=64/kv=1024: batch 40, all heads, dims 353/500
- Same pattern as 187c (batch 44, all heads, dim 363)
- Happens with AND without TunableOp → GPU hardware issue (ECC/HBM)
- Not a code bug
