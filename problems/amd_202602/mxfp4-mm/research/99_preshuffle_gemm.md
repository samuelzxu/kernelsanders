# #99 Preshuffle GEMM Variant

## Hypothesis
Use `gemm_afp4wfp4_preshuffle` for M>16 shapes. This variant:
- Takes B_shuffle (N//16, K*16) directly (optimized tile layout)
- Uses shuffle_scales format (N//32, K) for B_scale
- Has remap_xcd() like the standard variant
- Has tuned configs for N=2112-K=7168, N=3072-K=1536, N=4096-K=512

## Scale format
- e8m0_shuffle: (N, K//32) with CK-compatible permutation
- shuffle_scales: (N//32, K) with preshuffle-compatible permutation
- These are DIFFERENT permutations of the same data
- Need: e8m0_unshuffle → shuffle_scales conversion

## Extra overhead per call
- e8m0_unshuffle(B_scale_sh) + shuffle_scales(B_scale): ~1µs for B
- shuffle_scales(A_scale): ~0.5µs for A (M>=32 only)
- Total: 1-1.5µs extra

## Expected benefit
- Better L2 cache utilization from (N//16, K*16) tile layout
- Each 16-row N group has contiguous K data in L2

## Results
FAILED - two issues:
1. First attempt: KeyError 'float4_e2m1fn_x2' - B_shuffle in fp4x2 dtype not supported by Triton pointer canonicalization. Fixed with .view(torch.uint8).
2. Second attempt: NameError('b is not defined') - KNOWN preshuffle kernel JIT bug.
   The Triton compiler checks both branches of `if EVEN_K` even when EVEN_K is constexpr True.
   Variable `b` is defined inside `if EVEN_K:` but referenced outside. Dead branch raises error.
   This is a fundamental bug in the aiter preshuffle kernel code. Cannot be worked around
   without modifying the source file on the runner (which previously caused timeout issues).

CONCLUSION: Preshuffle variant is BLOCKED by JIT bug. Not usable.
