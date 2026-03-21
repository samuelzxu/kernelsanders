# Attempt 15: More splits for very large batches

## Changes
- bs <= 4: 16 splits
- bs 5-128: 32 splits
- bs > 128: 64 splits (NEW)

## Hypothesis
Very large batch sizes (bs=256) might benefit from more parallelism
with 64 splits instead of 32.

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

## Expected Impact
Only bs=256 cases should change (using 64 splits instead of 32).

## Results - Mixed (Slight Improvement)
| Batch | KV Len | Previous | New | Change |
|-------|--------|----------|-----|--------|
| 4 | 1024 | 56.6 | 56.7 | +0.2% |
| 4 | 8192 | 65.8 | 65.4 | -0.6% ✓ |
| 32 | 1024 | 64.0 | 64.4 | +0.6% |
| 32 | 8192 | 108 | 108 | same |
| 64 | 1024 | 73.2 | 73.6 | +0.5% |
| 64 | 8192 | 156 | 154 | -1.3% ✓ |
| 256 | 1024 | 122 | 123 | +0.8% |
| 256 | 8192 | 323 | 320 | -0.9% ✓ |

## Analysis
- bs=256/kv=8192: Slight improvement with 64 splits
- bs=256/kv=1024: Slight regression
- Other cases: Essentially unchanged (within variance)
- Overall: Marginal improvement, keeping this config
