# Attempt 30: Inlined Minimal Python Overhead

## Changes from Attempt 28
- Removed _run_stage1_reduce helper (inline calls directly)
- Tuple-based cache instead of dict-based (faster unpacking)
- Pre-bound _stage1 and _reduce function references at module level
- Shorter variable names to reduce bytecode overhead
- Removed a16w8 fallback (confirmed working, no try/except needed)

## Results
Marginal but consistent improvements (~0.5-1% on medium/large batches).
Geometric mean: ~40.5 µs (was ~41 µs)

## Code size: 75 lines total
