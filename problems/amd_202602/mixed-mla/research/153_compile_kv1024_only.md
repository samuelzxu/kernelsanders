# Attempt 153: torch.compile for kv<=1024 GEMM only

## Discovery from 152
- torch.compile improves kv=1024 GEMM: bs=4: -4.7µs, bs=32: -4.4µs
- But HURTS kv=8192 GEMM: bs=4: +5.4µs
- Also slight assembly regression (~6-10µs across kv>1024)

## Fix: restrict torch.compile to kv<=1024 GEMM
- kv<=1024 GEMM: compiled (fast, fewer kernel launches)
- kv>1024 GEMM (bs<=4 only): original baddbmm (uncompiled)
- Assembly: unchanged (no compilation interference)

## Expected Results
- bs=4/kv=1024: 21.7µs (from 152, -4.7µs vs 137)
- bs=4/kv=8192: 42.5µs (from 137, no regression)
- bs=32/kv=1024: 39.5µs (from 152, -4.4µs vs 137)
- Assembly configs: same as 137 (no interference)
- Geomean: ~73.8µs (3.8% improvement)
