# Attempt 5: asm_moe Investigation

## Hypothesis
The `asm_moe` function might be faster than `fused_moe` (up to 3x speedup reported).

## Research Findings
- `asm_moe` (specifically `rocm_aiter_asm_moe_tkw1_impl`) is optimized for **FP8 block-scale quantization**
- It is **NOT compatible with MXFP4** quantization
- MXFP4 support in vLLM uses different backends (Cutlass, FlashInfer)

## Conclusion
**Not applicable** - asm_moe doesn't support MXFP4 format required by this problem.
