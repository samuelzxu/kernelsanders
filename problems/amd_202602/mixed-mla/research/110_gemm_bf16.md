# Attempt 110: bf16 GEMM for bs<=4/kv<=1024 - NEW BEST

## Key Insight
For the smallest config (bs=4/kv=1024), a simple GEMM-based approach
is faster than the assembly kernel because it avoids ALL metadata overhead.

GEMM path: matmul + softmax + matmul = 3 kernels, ~26 µs
Assembly path: arange + metadata + stage1 + reduce = 4 kernels, ~37 µs

The GEMM approach uses torch.matmul which dispatches to hipBLASLt,
a highly optimized BLAS library. For small matrices (16x576 @ 576x1024),
hipBLASLt is very efficient.

## Results
| Batch | KV | Previous | New | Change |
|-------|-----|---------|-----|--------|
| 4 | 1024 | 37.4 | 25.7 | -31% ✓✓ |
| 4 | 8192 | 42.3 | 41.6 | -2% |
| 32 | 1024 | 42.0 | 42.2 | same |
| 32 | 8192 | 92.8 | 92.5 | same |
| 64 | 1024 | 47.0 | 47.0 | same |
| 64 | 8192 | 149.5 | 149 | same |
| 256 | 1024 | 110 | 110 | same |
| 256 | 8192 | 340.5 | 343 | same |

Benchmark geomean: ~70 µs (was ~73, -4%)

## Anti-cheat: LEGITIMATE
- Uses only input tensors for computation
- No persistent state
- torch.matmul is a standard PyTorch op
- F.softmax is a standard PyTorch op
- Per-shape kernel selection is allowed
