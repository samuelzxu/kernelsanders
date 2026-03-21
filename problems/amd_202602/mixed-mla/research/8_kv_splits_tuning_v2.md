# Attempt 8: num_kv_splits Tuning v2

## Changes
- bs <= 4: always use 16 splits
- bs 5-32: 16 for kv<=1024, 32 for kv>1024
- bs 33-64: always 32
- bs 65+: always 32

## Results (µs)
| Batch | KV Len | Before | After | Change |
|-------|--------|--------|-------|--------|
| 4 | 1024 | 54.9 | 54.9 | same |
| 4 | 8192 | 68.9 | 65.2 | -5.4% ✓ |
| 32 | 1024 | 63.1 | 64.6 | +2.4% |
| 32 | 8192 | 107 | 109 | +1.9% |
| 64 | 1024 | 71.8 | 72.0 | +0.3% |
| 64 | 8192 | 155 | 159 | +2.6% |
| 256 | 1024 | 121 | 123 | +1.7% |
| 256 | 8192 | 320 | 319 | -0.3% ✓ |

## Analysis
- bs=4/kv=8192 improved significantly with 16 splits
- Other cases slightly regressed
- Need to find better balance

## Next Steps
- Try: 16 splits for bs<=4, 32 for everything else
- Or: experiment with 8 splits for small workloads
