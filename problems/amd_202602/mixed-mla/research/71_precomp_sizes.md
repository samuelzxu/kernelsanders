# Attempt 71: Pre-computed metadata sizes

## Changes
- Compute metadata buffer sizes at module level using _CU_NUM hardware constant
- Skip get_mla_metadata_info_v1 and get_device_properties per call

## Results
Within noise of attempt 66. Savings of ~1 µs from skipping Python call
are lost in run-to-run variance.

Benchmark geomean: ~78 µs (same as attempt 66)

## Current legitimate best: ~78 µs benchmark, ~82 µs ranked

## Remaining bottleneck breakdown (per call):
- a16w8 path (kv<=1024): metadata ~10µs + kernel 20-80µs + reduce 5-8µs
- a8w8 path (kv>1024): Triton quant ~5µs + metadata ~10µs + kernel 40-300µs + reduce 5-8µs

The metadata computation (get_mla_metadata_v1 GPU kernel) and buffer allocations
dominate the non-kernel overhead. Without persistent caching, this is unavoidable.
