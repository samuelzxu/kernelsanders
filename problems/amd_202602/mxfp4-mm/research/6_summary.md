# MXFP4-MM Optimization Summary

## Best Result: Hybrid CK/Triton Approach

**Geometric Mean: ~20.1µs (baseline ~24.3µs) = 17% improvement**
**Ranked Benchmark Final: 15.8, 34.9, 14.5, 14.7, 24.4, 23.4 µs**

### Strategy
- K <= 512: Use Triton GEMM (`gemm_afp4wfp4`) - 29% faster
- K > 512: Use CK GEMM (`aiter.gemm_a4w4`) - maintains baseline

### Results Comparison
| M   | N    | K    | Baseline [µs] | Hybrid [µs] | Improvement |
|-----|------|------|---------------|-------------|-------------|
| 4   | 2880 | 512  | 20.9          | 15.6        | 25%         |
| 16  | 2112 | 7168 | 34.6          | 34.6        | 0%          |
| 32  | 4096 | 512  | 22.7          | 14.9        | 34%         |
| 32  | 2880 | 512  | 22.0          | 15.0        | 32%         |
| 64  | 7168 | 2048 | 24.7          | 24.7        | 0%          |
| 256 | 3072 | 1536 | 23.2          | 23.5        | -1%         |

## Key Findings

### Why CK is untuned for our shapes
- Logs show: "not found tuned config in CKGEMM or asmGEMM, will use default config!"
- CK kernels have pre-tuned configs for common vLLM shapes, but not ours

### Why Triton is faster for small K
- Better autotuning via `_get_config(M, N, K)`
- Less launch overhead for small matrices

### Why Triton is slower for large K
- Double quantization overhead (must re-quantize B for unshuffled scales)
- Quantization cost scales with K

### Failed Attempts
1. **uint8 view for B_q** - Still need to quantize B for scales (no savings)
2. **Pure Triton** - Slower for large K due to double quantization
3. **Code cleanup** - Minimal impact (overhead is in kernels, not Python)

## Future Optimization Ideas
1. Reverse e8m0_shuffle to get unshuffled B_scale (avoid double quant)
2. Custom Triton config with split-K for large K
3. Tune the K threshold (try K<=1024)
4. Pre-allocate output tensors
