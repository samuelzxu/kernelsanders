# Final Summary: 60 Experiments (88-148)

## Best: #148 at ~15.86µs ranked geomean
Key: LLVM O1 via copy+redirect + all #118 configs

## Improvement breakdown over baseline (~24µs):
1. Config injection (all 6 shapes): ~4µs
2. Fused quant+GEMM for M<=16 KSPLIT=1: ~2µs
3. BSN=64 for K=2048: ~1.2µs
4. MFMA 32x32 for M=256: ~0.5µs
5. num_stages=4 for K=2048: ~0.4µs
6. LLVM O1 optimization: ~0.1µs
Total: ~8.2µs improvement (24→15.86)
