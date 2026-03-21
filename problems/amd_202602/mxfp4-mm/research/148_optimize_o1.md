# #148 LLVM OPTIMIZE_O1

## Results (Ranked)
- M=4: 12.3µs (same)
- M=16: 20.9µs (slightly better, -0.2µs)
- M=32: 12.6-12.7µs (same)
- M=64: 19.6µs (better, -0.2µs)
- M=256: 19.7µs (better, -0.3µs)
- Geomean: ~15.86µs (vs #118's ~15.9µs)

## Analysis
O1 optimizes less aggressively than O2/O3/Os, reducing register pressure
in the FP4 GEMM kernel. Small but consistent improvement on large shapes.
This is the FIRST opt level that produces DIFFERENT (better) code than O3!

## Opt level summary:
| Level | Geomean | Effect |
|-------|---------|--------|
| O0 | ~32µs | 2x slower (no optimization) |
| **O1** | **~15.86µs** | **Slightly better than O3** |
| O2 | ~15.9µs | Same as O3 |
| O3 | ~15.9µs | Baseline |
| Os | ~15.9µs | Same as O3 |
