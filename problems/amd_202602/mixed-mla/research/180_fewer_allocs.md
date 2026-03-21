# Attempt 180: Fewer allocations in assembly path

## Changes vs 170
1. Unrolled metadata list comprehension into explicit torch.empty calls
   - Avoids Python list creation + iteration overhead
2. Merged lg+ls into single FP32 allocation
   - 8 allocs → 7 allocs (saves 1 HIP allocator call)
