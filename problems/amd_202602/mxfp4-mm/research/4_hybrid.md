# Attempt 4: Hybrid CK + Triton Approach

## Hypothesis
From Attempt 3, we found:
- Triton is ~30% faster for K=512 (small K)
- CK is better for K>512 (large K)

The slowdown for large K is due to double quantization of B.

## Strategy
1. Try viewing B_q as uint8 to reuse pre-computed quantization
2. Use hybrid: Triton for K<=512, CK for K>512
3. Or if uint8 view works, use Triton for all cases

## Key Shapes
- K=512: M=4,32,32 - use Triton
- K=1536: M=256 - use CK
- K=2048: M=64 - use CK
- K=7168: M=16 - use CK

## Result (Ranked Benchmark)
| M   | N    | K    | Hybrid [µs] | Baseline [µs] | Improvement |
|-----|------|------|-------------|---------------|-------------|
| 4   | 2880 | 512  | 15.4        | 20.9          | 26%         |
| 16  | 2112 | 7168 | 34.5        | 34.6          | ~same       |
| 32  | 4096 | 512  | 14.6        | 22.7          | 36%         |
| 32  | 2880 | 512  | 14.7        | 22.0          | 33%         |
| 64  | 7168 | 2048 | 24.8        | 24.7          | ~same       |
| 256 | 3072 | 1536 | 23.5        | 23.2          | ~same       |

**Geometric Mean: ~20.4µs (was ~24.3µs) - 16% overall improvement!**

## Analysis
- K=512 cases: Triton gives ~35% speedup
- K>512 cases: CK kernel maintains baseline performance
- No regression in any case!

## Next Steps
- Try viewing B_q as uint8 for Triton to eliminate double quant overhead
- This could improve large K cases too
