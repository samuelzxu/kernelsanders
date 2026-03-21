# v131: Triton MOE Pipeline Exploration

## Idea
Replace CK 2-stage GEMM with Triton MOE MXFP4 kernels for dense shapes.
Unlike CK, Triton kernel configs can be injected via JSON files.

## Pipeline
1. Sort tokens (AITER sorting - same as v103)
2. Quant hidden_states → MXFP4 (same)
3. Stage-1: `fused_moe_mxfp4_silu(a1, w1, ...)` (Triton + SiLU)
4. Requant intermediate → MXFP4
5. Stage-2: `fused_moe_mxfp4(intermediate, w2, ...)` (Triton)

## Key Questions
- What tensor formats does fused_moe_mxfp4_silu expect?
  - A: fp4x2 (quantized) or bf16?
  - B: what layout? (sorted shuffled weights or raw?)
  - A_scale, B_scale: per-tensor or block scale?
  - A_mx_scale, B_mx_scale: E8M0 microscales
- Does the Triton kernel work with pre-shuffled weights?
- Can we inject configs via JSON to AITER_TRITON_CONFIGS_PATH/moe/?

## Risk
- Triton kernels may be slower than CK without proper tuning
- Format mismatch between CK-shuffled weights and Triton expectations
- SiLU fusion in Triton vs CK may have different precision

## Status: EXPLORING
