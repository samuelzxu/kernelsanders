# Attempt 132: nks=64 for bs<=4/kv>1024 - maximize GPU utilization

## Problem Identified
For bs=4, kv=8192 (the small-batch long-sequence config):
- Current: nks=16 → 4*16=64 work items on MI355X (256 CUs)
- Only 25% GPU utilization! 192 CUs sit idle.
- Result: 41.7 µs (5.3x slower than theoretical bandwidth limit of 7.9 µs)

## Fix
- bs<=4, kv>1024: nks=64 → 4*64=256 work items = exactly 1 full wave on MI355X
- Each CTA handles 8192/64=128 tokens (same compute per CTA as before with smaller batch)
- bs>4: unchanged (nks=32, already has good occupancy)

## Expected Impact
With 4x more concurrent CTAs:
- Better HBM bandwidth utilization (more in-flight requests)
- Stage1 should improve from ~35 µs to ~12-15 µs
- Total bs=4/kv=8192 target: ~20-25 µs (vs current 41.7 µs)

## Other configs unchanged
| Config | Path | nks | Work items |
|--------|------|-----|-----------|
| bs=4, kv=1024 | GEMM | - | - |
| bs=4, kv=8192 | a8w8 | 64 (was 16) | 256 (was 64) |
| bs=32, kv=1024 | GEMM | - | - |
| bs=32, kv=8192 | a8w8 | 32 | 1024 |
| bs=64, kv=1024 | a16w8 | 8 | 512 |
| bs=64, kv=8192 | a8w8 | 32 | 2048 |
| bs=256, kv=1024 | a16w8 | 8 | 2048 |
| bs=256, kv=8192 | a8w8 | 32 | 8192 |
