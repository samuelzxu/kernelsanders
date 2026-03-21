# Attempt 1: Baseline Naive Implementation

## What we tried
- Original submission used Python loop over batches
- Used `torch._scaled_mm` for fp8 matmul
- Fell back to bf16 for softmax @ V computation

## Results
Very slow - not benchmarked but clearly worse than reference.

## Issues
- Python loop overhead
- Non-optimized kernel for fp8 matmul
- Extra memory traffic from bf16 fallback

## Conclusion
Need to use the optimized aiter kernel like the reference does.
