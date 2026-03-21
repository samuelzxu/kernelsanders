# Attempt 27: a16w8 Hybrid (bf16 Q + fp8 KV for large batches)

## Changes
- bs <= 64: bf16 Q + bf16 KV (a16w16 kernel)
- bs > 64: bf16 Q + fp8 KV (a16w8 kernel) with fallback to a8w8
- Discovered `mla_a16w8_qh16_m16x4_n16x1_coex0_mask1_ps.co` assembly kernel
- No Q quantization needed for any batch size!

## Key Insight
The aiter assembly kernels include an a16w8 variant that accepts bf16 Q + fp8 KV.
This gives us:
- No Q quantization overhead (saves 3-4 kernel launches)
- fp8 KV bandwidth savings (half the bandwidth of bf16 KV)
- Best of both worlds for large batches

## Results - NEW BEST
| Batch | KV Len | Previous | New | Change |
|-------|--------|----------|-----|--------|
| 4 | 1024 | 30.0 | 30.3 | ~same |
| 4 | 8192 | 29.5 | 29.8 | ~same |
| 32 | 1024 | 33.6 | 33.5 | ~same |
| 32 | 8192 | 33.5 | 33.6 | ~same |
| 64 | 1024 | 45.4 | 45.3 | ~same |
| 64 | 8192 | 45.5 | 45.7 | ~same |
| 256 | 1024 | 103 | 90.3 | -12.3% ✓ |
| 256 | 8192 | 103 | 91.1 | -11.6% ✓ |

Geometric mean: ~43 µs (was ~46 µs, 7% improvement)

## Why a16w8 is faster than a8w8 for bs=256
- a8w8 requires quantize_fp8(q) which launches 3-4 GPU kernels
- a16w8 skips Q quantization entirely
- fp8 KV bandwidth is the same in both cases
- Net savings: ~12 µs from eliminated quantization kernels
