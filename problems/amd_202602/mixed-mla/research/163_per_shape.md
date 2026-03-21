# Attempt 163: Per-shape compiled GEMM + kvg=32

## Changes vs 160
- Two separate compiled functions (no dynamic=True):
  - _compiled_gemm_short: for kv=1024 (bs=4, bs=32)
  - _compiled_gemm_long: for kv=8192 (bs=4 only)
- Each compiles with exact shape specialization (no guards)

## Results
- bs=4/kv=1024: 19.6µs (160: 21.4, -8.4%)
- bs=4/kv=8192: 40.1µs (160: 41.9, -4.3%)
- Assembly configs: unchanged (no regression!)
- Geomean: 71.3µs (160: 72.4, -1.5%)

## Why it works
- dynamic=False: each shape gets fully specialized compiled kernel
- No guard check overhead between shapes
- Two torch.compile wrappers ≈ same module overhead as one
