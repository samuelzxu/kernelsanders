# Baseline MXFP4-MM Submission

## Hypothesis
Establish baseline performance using the reference implementation which uses:
1. `dynamic_mxfp4_quant()` - Triton kernel for MXFP4 quantization
2. `e8m0_shuffle()` - Scale shuffling for CK kernel compatibility
3. `aiter.gemm_a4w4()` - CK (Composable Kernel) based FP4xFP4 GEMM

## Benchmark Shapes
All shapes are small-M (inference-style GEMMs):
| M   | N    | K    |
|-----|------|------|
| 4   | 2880 | 512  |
| 16  | 2112 | 7168 |
| 32  | 4096 | 512  |
| 32  | 2880 | 512  |
| 64  | 7168 | 2048 |
| 256 | 3072 | 1536 |

## Reference Timings (from task.yml)
| M   | N    | K    | time [µs] |
|-----|------|------|-----------|
| 4   | 2880 | 512  | 8.198     |
| 16  | 2112 | 7168 | 20.873    |
| 32  | 4096 | 512  | 9.462     |
| 32  | 2880 | 512  | 9.173     |
| 64  | 7168 | 2048 | 12.738    |
| 256 | 3072 | 1536 | 12.219    |

## Result (Ranked Benchmark)
| M   | N    | K    | Time [µs] | Reference [µs] | vs Ref |
|-----|------|------|-----------|----------------|--------|
| 4   | 2880 | 512  | 20.9      | 8.198          | 0.39x  |
| 16  | 2112 | 7168 | 34.6      | 20.873         | 0.60x  |
| 32  | 4096 | 512  | 22.7      | 9.462          | 0.42x  |
| 32  | 2880 | 512  | 22.0      | 9.173          | 0.42x  |
| 64  | 7168 | 2048 | 24.7      | 12.738         | 0.52x  |
| 256 | 3072 | 1536 | 23.2      | 12.219         | 0.53x  |

**Geometric Mean: ~23.6 µs (baseline is SLOWER than reference)**

## Key Observations from Logs
1. `"shape is M:X, N:X, K:X, not found tuned config in CKGEMM or asmGEMM, will use default config!"` - untuned shapes
2. Kernels used: `f4gemm_bf16_per1x32Fp4_BpreShuffle_192x128` and `f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128`
3. `log2_k_split` parameter available for split-K parallelism
4. The quantization step adds ~10-15µs overhead on top of GEMM

## Next Steps
- Try Triton-based `gemm_afp4wfp4` instead of CK kernel
- Try manually specifying `log2_k_split` for split-K
- Profile to understand quant vs GEMM overhead
