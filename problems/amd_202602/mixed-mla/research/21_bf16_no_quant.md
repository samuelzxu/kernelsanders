# Attempt 21: bf16 Q + bf16 KV (No Quantization)

## Changes
- Use bf16 KV cache (`kv_data["bf16"]`) instead of fp8
- Pass Q directly as bf16 (no quantization needed)
- No q_scale/kv_scale needed
- Uses a16w16 assembly kernel

## Key Insight
The FP8 quantization of Q launches 3-4 GPU kernels per call:
1. abs().amax() - reduction
2. clamp() + division - element-wise
3. to(FP8_DTYPE) - cast

Eliminating ALL of these saves significant per-call overhead, especially for
small batches where the quantization overhead is a large fraction of total time.

The tradeoff: bf16 KV is 2x the bandwidth of fp8 KV, so the attention kernel
itself is slower for large batches. But for small/medium batches, the
quantization savings dominate.

## Results - MASSIVE IMPROVEMENT (NEW BEST!)
| Batch | KV Len | Previous (fp8) | New (bf16) | Change |
|-------|--------|----------------|------------|--------|
| 4 | 1024 | 49.2 | 30.5 | -38% ✓ |
| 4 | 8192 | 48.5 | 30.0 | -38% ✓ |
| 32 | 1024 | 56.5 | 33.7 | -40% ✓ |
| 32 | 8192 | 55.7 | 34.0 | -39% ✓ |
| 64 | 1024 | 64.9 | 45.0 | -31% ✓ |
| 64 | 8192 | 65.5 | 44.9 | -31% ✓ |
| 256 | 1024 | 110 | 111 | +1% |
| 256 | 8192 | 108 | 111 | +3% |

Geometric mean: ~47 µs (was ~65 µs, 28% improvement!)

## Analysis
- Small batches (bs=4,32): 38-40% faster - quantization was dominant cost
- Medium batches (bs=64): 31% faster - still big savings
- Large batches (bs=256): ~same - bf16 2x bandwidth offsets quant savings
- bf16 kernel: `mla_a16w16_qh16_m16x4_n16x1_coex0_mask1_ps`
- Maximum error increased from 0.0 to ~7e-05 (within tolerance)
