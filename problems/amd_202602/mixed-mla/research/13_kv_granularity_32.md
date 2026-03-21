# Attempt 13: kv_granularity=32

## Changes
- Changed `kv_granularity=max(PAGE_SIZE, 16)` to `kv_granularity=32`

## Hypothesis
Larger KV granularity (32 vs 16) might reduce the number of work items
and improve memory coalescing for larger batch sizes.

## Previous Best (Attempt 10)
| Batch | KV Len | Time (us) |
|-------|--------|-----------|
| 4 | 1024 | 56.6 |
| 4 | 8192 | 65.8 |
| 32 | 1024 | 64.0 |
| 32 | 8192 | 108 |
| 64 | 1024 | 73.2 |
| 64 | 8192 | 156 |
| 256 | 1024 | 122 |
| 256 | 8192 | 323 |

## Results - Mixed
| Batch | KV Len | Previous | New | Change |
|-------|--------|----------|-----|--------|
| 4 | 1024 | 56.6 | 56.2 | -0.7% ✓ |
| 4 | 8192 | 65.8 | 65.7 | -0.2% ✓ |
| 32 | 1024 | 64.0 | 64.4 | +0.6% |
| 32 | 8192 | 108 | 115 | +6.5% ❌ |
| 64 | 1024 | 73.2 | 76.3 | +4.2% ❌ |
| 64 | 8192 | 156 | 166 | +6.4% ❌ |
| 256 | 1024 | 122 | 129 | +5.7% ❌ |
| 256 | 8192 | 323 | 331 | +2.5% ❌ |

## Analysis
- Small batches (bs<=4): slight improvement
- Larger batches: significant regression
- kv_granularity=32 is worse overall due to larger batches
- Reverting to kv_granularity=16
