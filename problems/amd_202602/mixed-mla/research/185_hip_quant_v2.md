# Attempt 185: HIP quant (single fused kernel) for Q fp8

## Discovery
Triton dynamic_per_tensor_quant_fp8_i8 uses TWO kernel launches:
1. _dynamic_per_tensor_quant_fp8_i8_kernel (compute scale via amax)
2. _static_per_tensor_quant_fp8_i8_kernel (apply scale)

HIP aiter.dynamic_per_tensor_quant uses ONE fused kernel.
Savings: ~2.5µs per kv>1024 call (1 fewer kernel launch + no zero-fill).

## Change vs 170
- Use `per_tensor_quant_hip` from aiter.ops.quant instead of Triton kernel
- Don't import Triton quant module (may reduce Triton init overhead)
