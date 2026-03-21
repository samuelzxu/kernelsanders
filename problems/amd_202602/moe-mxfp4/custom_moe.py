"""
Custom Triton MoE GEMM kernel with debug output.
Tests d_expert=2048 correctness issue.
"""
import torch
import triton
import triton.language as tl
import sys
from task import input_t, output_t

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, moe_sorting, get_inter_dim
from aiter.ops.triton.quant import dynamic_mxfp4_quant


@triton.jit
def _moe_gemm_kernel(
    a_ptr, a_scale_ptr,
    w_ptr, w_scale_ptr,
    sorted_ids_ptr, sorted_expert_ids_ptr,
    out_ptr,
    num_valid_tokens, topk, K, N, E,
    stride_w_e, stride_w_n, stride_w_k,
    stride_ws_row, stride_ws_col,
    stride_a_m, stride_a_k,
    stride_as_m, stride_as_k,
    stride_o_m, stride_o_n,
    block_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    SCALE_GROUP: tl.constexpr = 32

    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    expert_block = pid_m * BLOCK_M // block_size
    # Guard: sorted_expert_ids may have fewer entries than our grid covers
    # Clamp to last valid block to prevent OOB read
    max_expert_blocks = tl.cdiv(num_valid_tokens, block_size)
    safe_expert_block = tl.minimum(expert_block, max_expert_blocks - 1)
    safe_expert_block = tl.maximum(safe_expert_block, 0)
    expert_id = tl.load(sorted_expert_ids_ptr + safe_expert_block)
    # Clamp expert_id to valid range to prevent OOB weight access
    expert_id = tl.minimum(expert_id, E - 1)
    expert_id = tl.maximum(expert_id, 0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K // 2)

    token_ids = tl.load(sorted_ids_ptr + offs_m)
    m_valid = token_ids < num_valid_tokens
    safe_token_ids = tl.where(m_valid, token_ids, 0)
    orig_token = safe_token_ids // topk

    a_ptrs = a_ptr + orig_token[:, None] * stride_a_m + offs_k[None, :] * stride_a_k
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP)
    as_ptrs = a_scale_ptr + orig_token[:, None] * stride_as_m + offs_ks[None, :] * stride_as_k

    w_base = w_ptr + expert_id.to(tl.int64) * stride_w_e
    w_ptrs = w_base + offs_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n

    ws_base = w_scale_ptr + expert_id.to(tl.int64) * N * stride_ws_row
    ws_ptrs = ws_base + offs_n[:, None] * stride_ws_row + offs_ks[None, :] * stride_ws_col

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k_iters = tl.cdiv(K, BLOCK_K)

    n_valid = offs_n < N
    for ki in range(num_k_iters):
        a = tl.load(a_ptrs, mask=m_valid[:, None], other=0)
        a_scales = tl.load(as_ptrs, mask=m_valid[:, None], other=0)
        w = tl.load(w_ptrs, mask=n_valid[None, :], other=0)
        w_scales = tl.load(ws_ptrs, mask=n_valid[:, None], other=0)

        accumulator = tl.dot_scaled(a, a_scales, "e2m1", w, w_scales, "e2m1", accumulator)

        a_ptrs += (BLOCK_K // 2) * stride_a_k
        w_ptrs += (BLOCK_K // 2) * stride_w_k
        as_ptrs += (BLOCK_K // SCALE_GROUP) * stride_as_k
        ws_ptrs += (BLOCK_K // SCALE_GROUP) * stride_ws_col

    c = accumulator.to(tl.bfloat16)
    out_ptrs = out_ptr + offs_m[:, None] * stride_o_m + offs_n[None, :] * stride_o_n
    out_mask = m_valid[:, None] & (offs_n[None, :] < N)
    tl.store(out_ptrs, c, mask=out_mask)


def moe_gemm(a_q, a_scale, w, w_scale, sorted_ids, sorted_expert_ids,
             num_valid_tokens, topk, block_size, E, N, K):
    max_sorted = sorted_ids.shape[0]
    out = torch.empty((max_sorted, N), dtype=torch.bfloat16, device=a_q.device)

    BLOCK_M = block_size
    BLOCK_N = 128
    BLOCK_K = 128

    num_m_blocks = triton.cdiv(max_sorted, BLOCK_M)
    num_n_blocks = triton.cdiv(N, BLOCK_N)
    grid = (num_m_blocks * num_n_blocks,)

    w_u8 = w.view(torch.uint8)
    ws_u8 = w_scale.view(torch.uint8)

    _moe_gemm_kernel[grid](
        a_q, a_scale,
        w_u8, ws_u8,
        sorted_ids, sorted_expert_ids,
        out,
        num_valid_tokens, topk, K, N, E,
        w_u8.stride(0), w_u8.stride(1), w_u8.stride(2),
        ws_u8.stride(0), ws_u8.stride(1),
        a_q.stride(0), a_q.stride(1),
        a_scale.stride(0), a_scale.stride(1),
        out.stride(0), out.stride(1),
        block_size=block_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


_debug = [True]

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]
    d_expert_pad = config["d_expert_pad"]
    d_hidden = config["d_hidden"]

    M = hidden_states.shape[0]
    E = gate_up_weight.shape[0]
    topk = topk_ids.shape[1]

    # For large E or large d_expert, use fused_moe
    if E > 64 or d_expert_pad > 1024:
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

    # Custom Triton MoE
    N1 = gate_up_weight.shape[1]
    N2 = down_weight.shape[1]
    K1 = gate_up_weight.shape[2] * 2
    K2 = down_weight.shape[2] * 2
    _, model_dim, inter_dim = get_inter_dim(
        gate_up_weight_shuffled.shape, down_weight_shuffled.shape
    )
    dtype = hidden_states.dtype
    device = hidden_states.device
    block_m = 32

    if _debug[0]:
        _debug[0] = False
        w_u8 = gate_up_weight.view(torch.uint8)
        ws_u8 = gate_up_weight_scale.view(torch.uint8)
        print(f"[DEBUG] M={M} E={E} topk={topk} d_expert_pad={d_expert_pad}", file=sys.stderr)
        print(f"[DEBUG] N1={N1} K1={K1} N2={N2} K2={K2}", file=sys.stderr)
        print(f"[DEBUG] w shape={w_u8.shape} strides={w_u8.stride()}", file=sys.stderr)
        print(f"[DEBUG] ws shape={ws_u8.shape} strides={ws_u8.stride()}", file=sys.stderr)
        print(f"[DEBUG] E*N1={E*N1} scale_rows={ws_u8.shape[0]}", file=sys.stderr)

    # Sort
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m,
    )
    num_valid = M * topk

    # Quantize input
    h_q, h_scale = dynamic_mxfp4_quant(hidden_states)

    # Stage 1: gate_up GEMM
    gate_up_out = moe_gemm(
        h_q.view(torch.uint8), h_scale.view(torch.uint8),
        gate_up_weight, gate_up_weight_scale,
        sorted_ids, sorted_expert_ids,
        num_valid, topk, block_m,
        E, N1, K1,
    )

    # SwiGLU
    gate = gate_up_out[:, :d_expert_pad].float()
    up = gate_up_out[:, d_expert_pad:].float()
    intermediate = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)

    # Quantize intermediate
    inter_q, inter_s = dynamic_mxfp4_quant(intermediate)

    # Stage 2: down GEMM
    down_out = moe_gemm(
        inter_q.view(torch.uint8), inter_s.view(torch.uint8),
        down_weight, down_weight_scale,
        sorted_ids, sorted_expert_ids,
        num_valid, topk, block_m,
        E, N2, K2,
    )

    # Weighted reduction
    valid_mask = sorted_ids < num_valid
    valid_sids = sorted_ids[valid_mask]
    valid_weights = sorted_weights[valid_mask]
    valid_down = down_out[valid_mask, :d_hidden]
    token_indices = valid_sids // topk
    weighted = valid_weights.unsqueeze(1) * valid_down.float()
    output = torch.zeros((M, d_hidden), dtype=torch.float32, device=device)
    output.index_add_(0, token_indices.long(), weighted)

    return output.to(torch.bfloat16)
