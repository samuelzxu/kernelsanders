# Attempt 3: Triton-based MXFP4 GEMM

## Hypothesis
The CK kernel has no tuned config for our shapes ("not found tuned config in CKGEMM or asmGEMM").
The Triton-based `gemm_afp4wfp4` has its own autotuning via `_get_config()` and may perform better.

## Challenge
The Triton kernel expects UNSHUFFLED scales, but we only have B_scale_sh (shuffled).
Solution: Re-quantize B to get unshuffled scales.

## Changes
- Use `gemm_afp4wfp4` from `aiter.ops.triton.gemm.basic.gemm_afp4wfp4`
- Quantize B without shuffling to get proper scale format
- Use unshuffled A and B scales

## Result (Ranked Benchmark)
| M   | N    | K    | Triton [µs] | Baseline [µs] | Change |
|-----|------|------|-------------|---------------|--------|
| 4   | 2880 | 512  | 15.6        | 20.9          | -25%   |
| 16  | 2112 | 7168 | 37.4        | 34.6          | +8%    |
| 32  | 4096 | 512  | 15.0        | 22.7          | -34%   |
| 32  | 2880 | 512  | 14.9        | 22.0          | -32%   |
| 64  | 7168 | 2048 | 28.5        | 24.7          | +15%   |
| 256 | 3072 | 1536 | 24.2        | 23.2          | +4%    |

**Mixed results!** Triton faster for small K (512), slower for large K (7168, 2048).
The slowdown is due to re-quantizing B (double work for large K).

## Next Steps
- Hybrid approach: use Triton for K<=512, CK for K>512
- Or find way to reuse B_q without dtype conversion issue
