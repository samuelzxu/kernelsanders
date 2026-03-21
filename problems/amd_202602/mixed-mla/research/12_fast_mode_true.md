# Attempt 12: Enable fast_mode=True

## Changes
- Changed `fast_mode=False` to `fast_mode=True` in both:
  - `get_mla_metadata_info_v1()`
  - `get_mla_metadata_v1()`

## Hypothesis
The `fast_mode` flag might enable optimized metadata generation or kernel dispatch paths that reduce overhead.

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

## Results - MASSIVE REGRESSION
| Batch | KV Len | Previous | New | Change |
|-------|--------|----------|-----|--------|
| 4 | 1024 | 56.6 | 81.4 | +44% ❌ |
| 4 | 8192 | 65.8 | 94.3 | +43% ❌ |
| 32 | 1024 | 64.0 | 124 | +94% ❌ |
| 32 | 8192 | 108 | 174 | +61% ❌ |
| 64 | 1024 | 73.2 | 149 | +104% ❌ |
| 64 | 8192 | 156 | 235 | +51% ❌ |
| 256 | 1024 | 122 | 219 | +80% ❌ |
| 256 | 8192 | 323 | 418 | +29% ❌ |

## Analysis
`fast_mode=True` is MUCH slower (~1.5-2x worse across all cases).
This is counterintuitive but the "fast" mode seems to use a different
code path optimized for different workloads.

## Conclusion
Keep `fast_mode=False`. Reverted.
