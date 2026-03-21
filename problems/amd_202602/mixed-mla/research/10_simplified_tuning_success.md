# Attempt 10: Simplified Tuning - Success!

## Changes
Simplified num_kv_splits to:
- bs <= 4: 16 splits
- bs > 4: 32 splits

## Ranked Benchmark Results (µs)
| Batch | KV Len | Previous | New | Improvement |
|-------|--------|----------|-----|-------------|
| 4 | 1024 | 57.9 | 56.6 | -2.2% |
| 4 | 8192 | 67.9 | 65.8 | -3.1% |
| 32 | 1024 | 65.1 | 64.0 | -1.7% |
| 32 | 8192 | 115 | 108 | -6.1% |
| 64 | 1024 | 74.9 | 73.2 | -2.3% |
| 64 | 8192 | 165 | 156 | -5.5% |
| 256 | 1024 | 129 | 122 | -5.4% |
| 256 | 8192 | 332 | 323 | -2.7% |

## All tests passed
- 4/4 tests with Maximum error: 0.0

## Geometric Mean Improvement
Previous: [57.9, 67.9, 65.1, 115, 74.9, 165, 129, 332]^(1/8) ≈ 103.3 µs
New: [56.6, 65.8, 64.0, 108, 73.2, 156, 122, 323]^(1/8) ≈ 99.7 µs

Improvement: ~3.5% better geometric mean!

## vs Reference
| Case | Reference | Ours | Speedup |
|------|-----------|------|---------|
| bs=4, kv=1k | ~118 µs | 56.6 µs | 2.1x |
| bs=4, kv=8k | ~113 µs | 65.8 µs | 1.7x |
| bs=64, kv=8k | ~171 µs | 156 µs | 1.1x |
| bs=256, kv=8k | ~349 µs | 323 µs | 1.08x |

## Key Insight
Simple tuning is often better than complex heuristics.
The key optimization was:
1. Avoid CPU sync (2x+ speedup)
2. Use fewer splits for small batches (reduces reduction overhead)
