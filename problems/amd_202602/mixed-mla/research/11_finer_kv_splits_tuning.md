# Attempt 11: Finer-grained num_kv_splits Tuning

## Changes
- bs <= 4: 8 splits for kv<=2048, 16 for kv>2048
- bs 5-32: 16 splits
- bs > 32: 32 splits

## Hypothesis
Based on attempt 10's success with simplified tuning, we're now testing if:
1. Even fewer splits (8) helps for small batch + short kv
2. Medium batches (5-32) might be better with 16 splits instead of 32

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

Geometric mean: ~99.7 us

## Expected Impact
- bs=4/kv=1024: Should improve (8 splits vs 16)
- bs=32/kv=1024: Should improve (16 splits vs 32)
- bs=32/kv=8192: Should improve (16 splits vs 32)
- Larger batches: No change

## Results - REGRESSION
| Batch | KV Len | Previous | New | Change |
|-------|--------|----------|-----|--------|
| 4 | 1024 | 56.6 | 57.0 | +0.7% ❌ |
| 4 | 8192 | 65.8 | 65.9 | +0.2% |
| 32 | 1024 | 64.0 | 64.4 | +0.6% |
| 32 | 8192 | 108 | 114 | +5.6% ❌ |
| 64 | 1024 | 73.2 | 74.3 | +1.5% ❌ |
| 64 | 8192 | 156 | 165 | +5.8% ❌ |
| 256 | 1024 | 122 | 128 | +4.9% ❌ |
| 256 | 8192 | 323 | 330 | +2.2% ❌ |

## Analysis
- Using 8 splits for bs=4/kv<=2048 did NOT help
- Using 16 splits for bs=32 caused regression vs 32
- The previous simple approach (16 for bs<=4, 32 for rest) is better

## Conclusion
Reverting to attempt 10's configuration. Simple tuning wins.
