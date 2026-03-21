# Attempt 66: aiter Triton Quantization - NEW BEST

## Changes
- Use aiter's pre-compiled Triton quantization (dynamic_per_tensor_quant_fp8_i8)
- 2 kernel launches vs 3-4 for PyTorch quantization
- Still uses a16w8 for kv<=1024, a8w8 for kv>1024

## Results
| Batch | KV | Benchmark | Ranked | vs Previous |
|-------|-----|-----------|--------|-------------|
| 4 | 1024 | 38.8 | 41.5 | same |
| 4 | 8192 | 44.6 | 45.2 | -34% ✓ |
| 32 | 1024 | 42.7 | 44.9 | same |
| 32 | 8192 | 88.2 | 90.8 | -17% ✓ |
| 64 | 1024 | 48.3 | 51.9 | same |
| 64 | 8192 | 142 | 145 | -9% ✓ |
| 256 | 1024 | 110 | 113 | same |
| 256 | 8192 | 333 | 339 | +7% |

Benchmark geomean: ~78 µs (was ~88 µs, -11%)
Ranked geomean: ~82 µs

## Key Insight
aiter's Triton quantization kernel uses atomic_max for scale computation
(1 kernel) + static quantization (1 kernel) = 2 launches total.
PyTorch's approach: abs().amax().clamp() + division + clamp + cast = 3-4 launches.
The 1-2 fewer kernel launches save ~10-20 µs per call.

## Anti-cheat: LEGITIMATE
- Uses aiter library's pre-compiled Triton kernel (not custom)
- No persistent state
- Fresh computation every call
