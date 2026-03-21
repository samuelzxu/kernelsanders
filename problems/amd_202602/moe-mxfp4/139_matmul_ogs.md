# v139: matmul_ogs MOE GEMM Integration Plan

## Architecture
Replace CK 2-stage pipeline with Triton matmul_ogs (moe_gemm_a4w4) for dense shapes.

### Current CK Pipeline (5 kernel launches):
1. moe_sorting_opus_fwd (sort tokens)
2. fused_dynamic_mxfp4_quant_moe_sort (quant + sort scales)
3. ck_moe_stage1 (FP4 GEMM stage 1 + SiLU)
4. fused_dynamic_mxfp4_quant_moe_sort (requant intermediate)
5. ck_moe_stage2 (FP4 GEMM stage 2)

### New matmul_ogs Pipeline (3-4 kernel launches):
1. Construct RoutingData from topk_ids/topk_weights (CPU + 1 GPU kernel)
2. Quant hidden_states → MXFP4 (1 kernel, OR built into matmul_ogs)
3. moe_gemm_a4w4(x, w1, x_scales, w1_scales, routing_data, apply_swiglu=True) → stage1 (1 kernel)
4. moe_gemm_a4w4(inter, w2, inter_scales, w2_scales, routing_data) → final output (1 kernel)

## Key Challenges
1. **Bitmatrix construction**: Need to create from topk_ids (packed boolean matrix)
2. **Weight format**: Needs column-major [E, K//2, N] - check if raw weights match
3. **Scale format**: Needs microscale tensors matching tl.dot_scaled expectations
4. **Routing format**: Need gather_indx and scatter_indx from topk data
5. **Input quant**: matmul_ogs might handle quant internally via _mxfp4_quant_kernel

## Weight Format Analysis
- Our raw gate_up_weight: [E, 2*d_expert_pad, d_hidden_pad//2] fp4x2
- matmul_ogs expects w: [E, K//2, N] fp4x2, column-major (stride(-2)==1)
- K = d_hidden (7168), N = 2*d_expert (gate+up fused)
- So w shape should be [E, 3584, 2*d_expert] but our raw weights are [E, 2*d_expert_pad, 3584]
- Need to check if raw weights are already column-major or need transposing

## Status: PLANNING
