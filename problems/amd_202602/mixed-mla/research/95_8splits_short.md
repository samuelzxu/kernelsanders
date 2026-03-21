# Attempt 95: 8 splits for kv=1024 - NEW BEST

## Changes
- kv<=1024: num_kv_splits=8 (was 16 for bs<=4, 32 otherwise)
- kv>1024: unchanged (16 for bs<=4, 32 otherwise)

## Results
| Batch | KV | Previous | New | Change |
|-------|-----|---------|-----|--------|
| 4 | 1024 | 39 | 39 | same |
| 4 | 8192 | 45 | 45 | same |
| 32 | 1024 | 44 | 43 | -2% |
| 32 | 8192 | 95 | 89 | -6% |
| 64 | 1024 | 49 | 49 | same |
| 64 | 8192 | 151 | 142 | -6% |
| 256 | 1024 | 113 | 112 | same |
| 256 | 8192 | 344 | 333 | -3% |

Benchmark geomean: ~74 µs (was ~77, -4%)

## Why it helps
- Fewer splits → smaller metadata buffers → faster metadata computation
- Smaller allocations for kv=1024 leave allocator in better state
- 8 splits with 128 tokens/split is still enough parallelism for 1024 tokens
