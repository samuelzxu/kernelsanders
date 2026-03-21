# Attempt 22: Hybrid bf16/fp8

## Changes
- Use bf16 for batch_size < 128 (no quantization overhead)
- Use fp8 for batch_size >= 128 (lower bandwidth)
- Full buffer caching for both paths

## Results - NEW BEST
| Batch | KV Len | Time (µs) |
|-------|--------|-----------|
| 4 | 1024 | 30.0 |
| 4 | 8192 | 29.5 |
| 32 | 1024 | 33.6 |
| 32 | 8192 | 33.5 |
| 64 | 1024 | 45.4 |
| 64 | 8192 | 45.5 |
| 256 | 1024 | 103 |
| 256 | 8192 | 103 |

Geometric mean: ~46 µs

## Comparison with alternatives
- Pure bf16 (attempt 21): bs=256 was 111 µs, now 103 µs with fp8
- Pure fp8 (attempt 19): bs=4 was 49 µs, now 30 µs with bf16
- Hybrid gets best of both worlds
