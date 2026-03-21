# Attempt 10: Custom Triton Kernel for MoE MXFP4

## Goal
Implement a custom Triton kernel for MXFP4 MoE that can potentially beat AITER's fused_moe.

## MoE Computation Flow
```
For each token i, for each expert j in topk_ids[i]:
  1. gate = x_i @ W_gate_j.T    # [d_hidden] -> [d_expert]
  2. up = x_i @ W_up_j.T        # [d_hidden] -> [d_expert]
  3. intermediate = SiLU(gate) * up  # SwiGLU activation
  4. expert_out = intermediate @ W_down_j.T  # [d_expert] -> [d_hidden]
  5. output_i += topk_weights[i,j] * expert_out
```

## MXFP4 Quantization
- FP4 (E2M1) values packed as fp4x2 (2 values per byte)
- E8M0 scales (block size = 32 elements)
- Per 1x32 block scaling

## Implementation Strategy

### Option A: Use AITER building blocks
1. Use `dynamic_mxfp4_quant` for activation quantization
2. Use `gemm_afp4wfp4_preshuffle` for GEMMs
3. Implement custom token sorting and reduction
4. Implement SwiGLU fusion

### Option B: Full custom Triton
1. Implement MXFP4 GEMM from scratch
2. Implement token routing
3. Implement fused SwiGLU
4. Implement weighted reduction

### Option C: Hybrid approach
1. Use AITER for quantization
2. Custom Triton for expert-parallel GEMM
3. Fuse operations where possible

## Key Optimizations to Target
1. Better memory coalescing for expert routing
2. Fused SwiGLU activation
3. Fused weighted reduction
4. Split-K for large d_expert

## Implementation Plan
1. Start with Option A (easiest)
2. Profile to identify bottlenecks
3. Implement custom kernels for bottlenecks
4. Iterate

## Status
Starting implementation...
