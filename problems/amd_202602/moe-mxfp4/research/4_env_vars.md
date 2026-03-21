# Attempt 4: Environment Variables

## Hypothesis
Setting AITER-related environment variables before import might enable additional optimizations:
- `AITER_MOE=1`
- `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1`

These might trigger different kernel selection or enable assembly-optimized paths.

## Code change
Add environment variable settings before AITER import.

## Result (Ranked Benchmark)
| bs | E | d_expert | Time [µs] | Baseline [µs] | Diff |
|----|---|----------|-----------|---------------|------|
| 16 | 257 | 256 | 137 | 131 | +4.6% |
| 128 | 257 | 256 | 222 | 218 | +1.8% |
| 512 | 257 | 256 | 252 | 251 | +0.4% |
| 16 | 33 | 512 | 95.4 | 90.5 | +5.4% |
| 128 | 33 | 512 | 132 | 128 | +3.1% |
| 512 | 33 | 512 | 217 | 215 | +0.9% |
| 512 | 33 | 2048 | 349 | 346 | +0.9% |

## Conclusion
Environment variables made performance **WORSE** (1-5% slower across all cases).
The baseline configuration is already optimal. Reverting to baseline.
