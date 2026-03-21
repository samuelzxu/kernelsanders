# Attempt 115: baddbmm + FP32 softmax - BEST

## Changes from 112
- torch.baddbmm(beta=0, alpha=SM_SCALE) fuses scale into GEMM
- Saves 1 element-wise kernel (scale multiplication)
- FP32 softmax maintained for correctness

## Results
| Batch | KV | Method | Best Run |
|-------|----|--------|----------|
| 4 | 1024 | GEMM | 24.1 µs |
| 4 | 8192 | a8w8 | 41.5 µs |
| 32 | 1024 | GEMM | 39.5 µs |
| 32 | 8192 | a8w8 | 88.6 µs |
| 64 | 1024 | a16w8 | 46.2 µs |
| 64 | 8192 | a8w8 | 143 µs |
| 256 | 1024 | a16w8 | 108 µs |
| 256 | 8192 | a8w8 | 333 µs |

Best benchmark geomean: ~65 µs
Average across runs: ~67 µs

## Architecture
- bs<=32/kv<=1024: GEMM (baddbmm + softmax + bmm) - 3 kernels
- kv<=1024/bs>32: a16w8 assembly - 4 kernels
- kv>1024: a8w8 + Triton quant - 6 kernels
