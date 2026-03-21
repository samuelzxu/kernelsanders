# #83 Hybrid a16wfp4 (K≤512) + Triton (K>512)

## Hypothesis
gemm_a16wfp4 was catastrophically slow for large K (#81: 40.9µs for K=7168)
because the on-the-fly quant loop runs many iterations. But for K=512:
- Loop iterations: K/BSK = 512/512 = 1 (or 512/256 = 2)
- The quant overhead is minimal
- Saves dynamic_mxfp4_quant(A) entirely (~3µs)

For K=512 shapes specifically: gemm_a16wfp4 = quant_B + 1 kernel (fused)
vs #53 = quant_A + quant_B + GEMM (3 kernels) or fused(quant_A+GEMM) + quant_B (2 kernels)

## Expected
- K=512 shapes: ~10-11µs (save quant_A launch overhead)
- Large K shapes: unchanged from #53
