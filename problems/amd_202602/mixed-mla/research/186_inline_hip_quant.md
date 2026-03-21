# Attempt 186: Inline HIP quant via aiter.dynamic_per_tensor_quant

## Changes vs 185
- Call aiter.dynamic_per_tensor_quant directly (no extra module import)
- Don't import aiter.ops.quant or aiter.ops.triton.quant.quant
- Lighter module initialization, potentially less GPU state pollution
