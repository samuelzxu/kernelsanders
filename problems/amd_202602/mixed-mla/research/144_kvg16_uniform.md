# Attempt 144: kvg=16 uniform + a16w8 for all + nks=32 uniform

## Key Discovery
The reference implementation uses `kv_granularity=16` and `num_kv_splits=32` for ALL configs.
We've been using kvg=64 for kv>1024 and nks=8 for kv<=1024.

Was kvg=64 ever properly A/B tested? Maybe kvg=16 is better because:
- More fine-grained metadata = better load balancing across CTAs
- The assembly kernel was tuned/designed for kvg=16
- More groups per CTA = more independent work items = better utilization

## Changes vs 143
- kv>1024: kvg=16 (was 64) — match reference
- kv<=1024: nks=32 (was 16) — match reference
- All assembly: a16w8 (same as 143, skip Q quantization)

## Key hypothesis
Our kvg=64 for kv>1024 was adopted to speed up metadata computation.
But metadata is a tiny fraction of total time. The actual attention kernel
might perform BETTER with kvg=16 due to better load balancing.

## Expected Impact
If kvg=16 improves kernel efficiency by even 2-3%:
- bs=32/kv=8192: 88.6 → ~86µs
- bs=64/kv=8192: 141 → ~137µs
- bs=256/kv=8192: 338 → ~328µs
Combined with Q quant savings: potentially 73-75µs geomean
