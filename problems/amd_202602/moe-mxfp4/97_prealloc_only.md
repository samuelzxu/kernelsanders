# v97: Pre-allocated Sorting Buffers Only

## Approach
Same as v95 but WITHOUT the block_m=32 override for E=33 d=512 shapes.
v95 showed 157µs geomean (worse than v85's ~153µs), likely due to
the block_m=32 override hurting E=33 d=512 performance.

This isolates the effect of buffer pre-allocation alone.

## Changes from v85
- Added `_fast_sorting()` that caches sorting buffers and calls
  `aiter.moe_sorting_fwd` directly (bypassing 5 torch.empty() calls)
