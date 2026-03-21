# #77 Custom Minimal Triton GEMM

## Hypothesis
A stripped-down Triton kernel with no XCD remapping, no split-K branching,
no EVEN_K checks might compile to tighter GPU code. The aiter kernel has
overhead from generic features we don't need for single-split shapes.

## Changes from aiter kernel
1. No `remap_xcd()` — direct PID mapping
2. No `EVEN_K` heuristic — always loads without masking (all our K values are divisible)
3. No split-K code path — only handles NUM_KSPLIT=1
4. No `tl.assume()` calls
5. Simplified pointer arithmetic
6. Uses `pid_grid()` for L2 grouping (kept — this is genuinely helpful)

## Shapes affected
- M=32/64/256 with K<=1024 (NUM_KSPLIT=1): uses custom kernel
- M=16 with K=7168 (NUM_KSPLIT=8): falls back to aiter's split-K kernel
- M<=16 with K<=512 (NUM_KSPLIT=1): uses fused quant+GEMM
