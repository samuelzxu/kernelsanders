# Attempt 28: a16w8 for bs >= 64

## Changes
- bs <= 32: bf16 Q + bf16 KV (a16w16)
- bs > 32: bf16 Q + fp8 KV (a16w8)

## Results - NEW BEST
| Batch | KV Len | Time (µs) |
|-------|--------|-----------|
| 4 | 1024 | 30.1 |
| 4 | 8192 | 29.7 |
| 32 | 1024 | 34.1 |
| 32 | 8192 | 34.1 |
| 64 | 1024 | 40.3 |
| 64 | 8192 | 40.5 |
| 256 | 1024 | 91.7 |
| 256 | 8192 | 91.6 |

Geometric mean: ~41 µs

## Key Finding
- a16w8 (bf16 Q + fp8 KV) is better than a16w16 (bf16 Q + bf16 KV) for bs=64
- fp8 KV bandwidth savings help at bs=64 (64 * kv_len * 576 bytes is significant)
- For bs=32, a16w16 is still better (data fits in cache)
- Threshold: use a16w16 for bs <= 32, a16w8 for bs > 32
