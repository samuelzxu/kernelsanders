# #79 PyTorch-based B_scale computation

## Hypothesis
For K=512: `dynamic_mxfp4_quant(B)` launches a Triton kernel to compute both
FP4 data AND scale. We only need the scale. Computing it with PyTorch ops
(view + abs + amax + log2 + floor + clamp) might be faster than launching a
Triton kernel because:
1. No kernel launch overhead
2. PyTorch ops are fused by the HIP backend
3. The computation is simple (just amax + log2 per block of 32)

## Risk
PyTorch ops might launch multiple small kernels (abs, amax, log2 each) which
could be slower than one Triton kernel.
