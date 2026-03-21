# #116 Scale-Only Kernel for K=512 B_scale

## Approach
For K=512 shapes, `dynamic_mxfp4_quant(B)` does FULL quantization (FP4 packing
+ scale computation) just to get B_scale. We throw away the FP4 result since
B_q is already provided.

Custom `_compute_e8m0_scale_kernel`: computes ONLY max-abs per 32-element
group → E8M0 encoding. ~1/3 the compute work of full quant.

## E8M0 encoding
E8M0 is the biased exponent of the max-abs value in fp32 representation:
bits = reinterpret_cast<int>(max_abs_f32)
exponent = (bits >> 23) & 0xFF  # biased exponent (0-255)

## Expected savings
For K=512 shapes (M=4, M=32): saves ~0.5-1µs per call by avoiding
unnecessary FP4 packing of the B matrix.

## Results
First attempt: FAILED correctness - E8M0 encoding was wrong. Raw biased exponent
doesn't match the actual MX scale formula (needs rounding up + subtract 2).

Second attempt (fixed formula): PASSED correctness but K=512 shapes are 6µs slower
(18µs vs 12µs). The new Triton kernel requires JIT compilation (~6µs overhead on
first call). `dynamic_mxfp4_quant`'s kernel is already cached from warmup.

## Key insight: Triton JIT compilation overhead
Each unique @triton.jit kernel specialization costs ~3-6µs for first-call
compilation. Adding ANY new Triton kernel to the submission risks degrading
first-call performance. The benchmark's warmup may not cover all specializations.

The only way to avoid JIT overhead: use EXISTING compiled kernels (from aiter)
or compile during module load (before benchmark starts).
