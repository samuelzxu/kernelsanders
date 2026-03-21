# Attempt 147: Micro-optimizations to reduce fixed overhead

## Changes vs 137
1. Merge lg+ls into single FP32 allocation (saves 1 GPU alloc = ~1µs)
2. Inline Q quantization (avoid Python function call overhead)
3. Remove intermediate `v` variable in GEMM path

## Rationale
Small-batch configs have 2x+ overhead vs theoretical minimum.
Fixed overhead (allocations, kernel launches) dominates for these configs.
Every 1µs saved helps the geomean.

## Expected Impact
- ~1-2µs improvement per assembly call from merged allocation
- Total geomean improvement: ~0.5-1% (marginal but real)
