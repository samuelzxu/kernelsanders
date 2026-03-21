# Attempt 7: Leaderboard Submission

## Results

### Tests
All 4/4 tests passed with Maximum error: 0.0

### Ranked Benchmark Results (µs)
| Batch | KV Len | Mean | Min | Max |
|-------|--------|------|-----|-----|
| 4 | 1024 | 57.9 | 55.9 | 61.7 |
| 4 | 8192 | 67.9 | 66.1 | 73.4 |
| 32 | 1024 | 65.1 | 63.0 | 70.9 |
| 32 | 8192 | 115 | 111 | 123 |
| 64 | 1024 | 74.9 | 72.6 | 79.1 |
| 64 | 8192 | 165 | 161 | 172 |
| 256 | 1024 | 129 | 126 | 134 |
| 256 | 8192 | 332 | 324 | 343 |

### Geometric Mean Calculation
Times: [57.9, 67.9, 65.1, 115, 74.9, 165, 129, 332]
Geometric mean = (57.9 × 67.9 × 65.1 × 115 × 74.9 × 165 × 129 × 332)^(1/8)
≈ 105.7 µs

### Comparison to Reference (from README)
| Case | Reference | Ours | Speedup |
|------|-----------|------|---------|
| bs=4, kv=1k | ~118 µs | 57.9 µs | 2.0x |
| bs=4, kv=8k | ~113 µs | 67.9 µs | 1.7x |
| bs=64, kv=8k | ~171 µs | 165 µs | 1.04x |
| bs=256, kv=8k | ~349 µs | 332 µs | 1.05x |

## Key Optimizations Applied
1. Use aiter mla_decode_fwd (fp8 Q + fp8 KV)
2. Avoid CPU sync by using tensor.shape instead of .item()
3. Dynamic num_kv_splits tuning

## Status
Successfully submitted to leaderboard. Waiting for ranking.
