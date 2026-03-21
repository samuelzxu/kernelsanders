# #96 Hybrid Fused (separate kernels for KSPLIT=1 vs KSPLIT>1)

## Hypothesis
Use old fused kernel for KSPLIT=1 (M=4), new fused+splitK+xcd for KSPLIT>1 (M=16).
Avoids extra JIT overhead on M=4 path.

## Results (Ranked)
- M=4: 14.0µs (STILL WORSE, +1.6µs - extra kernel defs increase module load)
- M=16: 19.7µs (better, -1.4µs)
- M=32: 12.3µs (slightly better)
- M=64: 20.4µs (same)
- M=256: 20.6µs (same)
- Geomean: ~16.0µs (same as #92)

## Conclusion
Having extra @triton.jit functions in the module adds ~1.6µs to first-call overhead,
even if those functions aren't called. The M=16 gain is offset by M=4 loss.
Config tuning has reached diminishing returns. Need fundamentally different kernel.
