# MXFP4-MM Optimization Results Summary

## Reference Timings (aiter.gemm_a4w4 with tuned configs)
| Shape | Ref (µs) |
|-------|----------|
| M=4, N=2880, K=512 | 8.2 |
| M=16, N=2112, K=7168 | 20.9 |
| M=32, N=4096, K=512 | 9.5 |
| M=32, N=2880, K=512 | 9.2 |
| M=64, N=7168, K=2048 | 12.7 |
| M=256, N=3072, K=1536 | 12.2 |
| **Geomean** | **~11.5** |

## Best Submission: #19 Hybrid Unshuffle (Triton)
| Shape | Time (µs) | vs Ref |
|-------|-----------|--------|
| M=4, N=2880, K=512 | 15.8 | 1.93x |
| M=16, N=2112, K=7168 | 30.5 | 1.46x |
| M=32, N=4096, K=512 | 14.9 | 1.57x |
| M=32, N=2880, K=512 | 14.7 | 1.60x |
| M=64, N=7168, K=2048 | 21.2 | 1.67x |
| M=256, N=3072, K=1536 | 21.8 | 1.79x |
| **Geomean** | **~19.1** | **~1.66x** |

## Approach
- K <= 512: Triton `gemm_afp4wfp4` with re-quantized B (double quant)
- K > 512: Triton `gemm_afp4wfp4` with `e8m0_unshuffle` to reuse pre-computed B_q

## Key Findings
1. **e8m0_unshuffle**: The correct inverse permutation is (0,5,3,1,4,2), not (0,5,3,4,2,1)
2. **Unshuffle vs Double Quant**: For K<=512, re-quantizing B is ~2µs faster than unshuffle. For K>=1536, unshuffle saves 2-7µs.
3. **CK vs Triton**: Triton is faster for all shapes when unshuffle is available. CK (aiter.gemm_a4w4) suffers from "not found tuned config" on the runner.
4. **Preshuffle kernel**: Has a Triton compilation bug (NameError: b is not defined) when EVEN_K is False.
5. **Shape-specific tuned configs**: Available in aiter repo but didn't improve over defaults on the runner.
6. **Pre-allocating output**: No measurable improvement.

## Failed Approaches
- Incorrect e8m0_unshuffle (wrong permutation): All tests fail
- gemm_afp4wfp4_preshuffle: Triton compilation bug
- Custom Triton configs: Missing required parameters (num_warps, etc.)
- aiter.gemm_a4w4 with log2_k_split: Not supported by high-level API
- gemm_afp4wfp4_preshuffled_scales for M>=32: Scale format mismatch

## Potential Further Optimizations
1. Write a custom Triton kernel optimized for these specific shapes
2. Find a way to provide tuned CK configs on the runner
3. Fuse quantization and GEMM into a single kernel
4. Optimize the unshuffle operation (avoid contiguous copy)
