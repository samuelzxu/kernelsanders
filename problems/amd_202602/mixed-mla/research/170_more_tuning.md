# Attempt 170: Increased TunableOp tuning budget

## Changes vs 168
- PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS=100
- PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS=50

## Results: NEW BEST 70.4µs (168 was 70.7µs)
| Config | 168 | 170 |
|--------|-----|-----|
| bs=4/kv=1024 | 19.9 | 19.5 |
| bs=4/kv=8192 | 38.7 | 39.4 |
| bs=32/kv=1024 | 33.7 | 33.1 |
| bs=32/kv=8192 | 89.3 | 88.6 |
| bs=64/kv=1024 | 50.2 | 50.6 |
| bs=64/kv=8192 | 142 | 140 |
| bs=256/kv=1024 | 113 | 113 |
| bs=256/kv=8192 | 339 | 339 |
