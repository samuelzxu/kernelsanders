# Attempt 5: View B_q as uint8 for Triton

## Hypothesis
The fp4x2 dtype (float4_e2m1fn_x2) is just 2 FP4 values packed in 1 byte - same storage as uint8.
If we view B_q as uint8, we can use pre-computed quantization with Triton GEMM.

This could give us:
- Triton's better small-K performance
- Without double quantization overhead for large K
- Best of both worlds!

## Challenge
Still need unshuffled B_scale. Options:
1. Quantize B just for scale (discard quantized tensor)
2. Store unshuffled scale in generate_input (not possible - can't modify)
3. Reverse e8m0_shuffle (need to understand the shuffle pattern)

## Result (Ranked Benchmark)
| M   | N    | K    | uint8+Triton [µs] | Hybrid [µs] | Change |
|-----|------|------|-------------------|-------------|--------|
| 4   | 2880 | 512  | 15.8              | 15.4        | +3%    |
| 16  | 2112 | 7168 | 38.5              | 34.5        | +12%   |
| 32  | 4096 | 512  | 15.5              | 14.6        | +6%    |
| 32  | 2880 | 512  | 15.2              | 14.7        | +3%    |
| 64  | 7168 | 2048 | 29.7              | 24.8        | +20%   |
| 256 | 3072 | 1536 | 24.5              | 23.5        | +4%    |

**Conclusion**: uint8 view doesn't help - we still need to quantize B for the scale.
The hybrid approach (Attempt 4) remains the best at ~20.4µs geomean (16% improvement).

## Analysis
- The view(uint8) works but doesn't save computation
- Still need `dynamic_mxfp4_quant(B)` to get unshuffled scale
- Triton is slower than CK for large K even with tuned configs
