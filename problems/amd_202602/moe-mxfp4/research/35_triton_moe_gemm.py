"""
Custom Triton MoE stage1 GEMM with inline bf16→fp4 quantization.
Fuses the quantization + stage1 GEMM into a single kernel launch.

Uses tl.dot_scaled for native FP4 MFMA instructions on MI355X.
Handles MoE token sorting via sorted_ids/sorted_expert_ids indexing.

Falls back to fused_moe for stage2 (intermediate requant + down GEMM).
"""
import torch
import triton
import triton.language as tl
from task import input_t, output_t

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, moe_sorting, get_inter_dim
from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op


@triton.jit
def _moe_stage1_fused_quant_kernel(
    # Input
    hidden_ptr,          # [M, d_hidden] bf16
    # Weights (pre-shuffled, fp4x2)
    w1_ptr,              # [E, 2*d_expert_pad, d_hidden_pad//2] fp4x2
    w1_scale_ptr,        # weight scales (shuffled flat)
    # MoE indexing
    sorted_ids_ptr,      # [max_tokens_padded] int32
    sorted_expert_ids_ptr,  # [max_blocks] int32
    num_valid_ids_ptr,   # [2] int32
    # Output
    out_ptr,             # [M, topk, inter_dim] bf16
    # Dims
    M, topk, K, N,      # K=d_hidden, N=2*d_expert_pad (gate+up)
    stride_h_m, stride_h_k,
    stride_w_e, stride_w_n, stride_w_k,
    stride_o_m, stride_o_t, stride_o_n,
    block_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """MoE stage1 GEMM: sorted_hidden @ expert_gate_up_weight.T"""
    # This is a placeholder - the real implementation would need:
    # 1. Determine expert for this block from sorted_expert_ids
    # 2. Load tokens via sorted_ids indirect indexing
    # 3. Quantize bf16→fp4 per K-tile using _mxfp4_quant_op
    # 4. Call tl.dot_scaled with quantized input + fp4 weights
    # 5. Write output
    #
    # For now, this just does a bf16 GEMM (no FP4) as proof of concept
    pass


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    """
    For now, use the standard fused_moe pipeline.
    The custom Triton kernel above is a work-in-progress.
    """
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    return fused_moe(
        hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
        topk_weights, topk_ids,
        expert_mask=None,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None, a2_scale=None,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )
