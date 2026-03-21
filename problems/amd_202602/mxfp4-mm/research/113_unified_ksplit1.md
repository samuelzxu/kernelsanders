# #113 Fused Quant + remap_xcd for KSPLIT=1 Shapes

## Results
CATASTROPHIC regression:
- M=64 K=2048: 53.7µs (was 20µs) - fused quant loads 4x more A data (bf16 vs fp4)
- M=256 K=1536: 193µs (was 20µs) - unknown interference from extra kernel def
- M=4, M=16: OK (12-20µs)
- M=32: 15.5µs (was 12.6µs)

## Analysis
Fused quant is ONLY beneficial for M<=16 where A data is tiny.
For M>=32, separate quant (fp4 output) + GEMM (fp4 input) is much faster
because the GEMM reads 4x less A data than the fused kernel.
The fused kernel's advantage (saving quant launch ~2µs) is overwhelmed
by reading 4x more A data in the GEMM inner loop.

## Conclusion
Fused quant only helps M<=16. For M>=32, always use separate quant + GEMM.
This is because:
- M=4: A = 4*512*2 = 4KB bf16 vs 4*256 = 1KB fp4 → 3KB extra (negligible)
- M=64: A = 64*2048*2 = 256KB bf16 vs 64*1024 = 64KB fp4 → 192KB extra per K-iter
