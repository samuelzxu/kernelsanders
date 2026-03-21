# Attempt 75: Tuned kv_granularity - MARGINAL IMPROVEMENT

## Changes
- kv_granularity=16 for kv<=1024 (standard)
- kv_granularity=64 for kv>1024 (coarser, faster metadata)

## Results
kv=8192 cases improved 2-3%, kv=1024 stayed same.
Benchmark geomean: ~77 µs (was ~78, -1.5%)

## Anti-cheat: LEGITIMATE
Per-shape configuration selection. No persistent state.
