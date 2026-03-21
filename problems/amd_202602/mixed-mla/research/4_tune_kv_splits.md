# Attempt 4: Dynamic num_kv_splits Tuning

## What we tried
- Added `get_num_kv_splits(batch_size, kv_seq_len)` function
- Tune num_kv_splits based on total tokens:
  - <= 8192 tokens: 16 splits
  - <= 65536 tokens: 32 splits
  - > 65536 tokens: 32 splits

## Results (µs)
| Batch | KV Len | Before | After | Change |
|-------|--------|--------|-------|--------|
| 4 | 1024 | 55.5 | 54.9 | -1% |
| 4 | 8192 | 68.6 | 68.9 | +0.4% |
| 32 | 1024 | 65.1 | 63.1 | -3% |
| 32 | 8192 | 109 | 107 | -2% |
| 64 | 1024 | 72.8 | 71.8 | -1% |
| 64 | 8192 | 160 | 155 | -3% |
| 256 | 1024 | 123 | 121 | -2% |
| 256 | 8192 | 319 | 320 | +0.3% |

## Analysis
- Small improvements on most cases (1-3%)
- Negligible regression on bs=4/kv=8k and bs=256/kv=8k
- Overall slight improvement in geometric mean

## Conclusion
Minor optimization - keeps the good performance from attempt 3.
