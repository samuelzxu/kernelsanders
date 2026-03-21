# Attempt 117: Clean baddbmm GEMM - BEST (~64 µs)

## Architecture
- bs<=32/kv<=1024: bf16 GEMM (baddbmm + softmax + bmm)
- kv<=1024/bs>32: a16w8 assembly (persistent)
- kv>1024: a8w8 + Triton quant (persistent)

## Best benchmark results
| Batch | KV | Time | Method |
|-------|----|------|--------|
| 4 | 1024 | 23.8 µs | GEMM |
| 4 | 8192 | 41.7 µs | a8w8 |
| 32 | 1024 | 38.9 µs | GEMM |
| 32 | 8192 | 86.5 µs | a8w8 |
| 64 | 1024 | 46.6 µs | a16w8 |
| 64 | 8192 | 139 µs | a8w8 |
| 256 | 1024 | 110 µs | a16w8 |
| 256 | 8192 | 334 µs | a8w8 |

Geometric mean: ~64 µs
Improvement over reference: 36%

## 117 attempts total
