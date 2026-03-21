# Attempt 6: Ensure Contiguous Tensors

## Hypothesis
Ensuring input tensors are contiguous may improve memory access patterns and performance.

## Code change
Add `.contiguous()` calls on input tensors before passing to fused_moe.

## Result (Ranked Benchmark)
| bs | E | d_expert | Time [µs] | Previous [µs] |
|----|---|----------|-----------|---------------|
| 16 | 257 | 256 | 131 | 138 |
| 128 | 257 | 256 | 215 | 222 |
| 512 | 257 | 256 | 249 | 252 |
| 16 | 33 | 512 | 90.9 | 96.2 |
| 128 | 33 | 512 | 129 | 132 |
| 512 | 33 | 512 | 215 | 217 |
| 512 | 33 | 2048 | 347 | 350 |

## Conclusion
Performance is similar to baseline, within run-to-run variance (~3-5%).
The contiguous() call is good practice but doesn't significantly impact performance
since benchmark inputs are already contiguous.
