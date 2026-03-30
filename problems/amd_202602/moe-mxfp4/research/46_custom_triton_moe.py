"""
Custom Triton MoE GEMM kernel using tl.dot_scaled for native FP4 MFMA.
Replaces expert-parallel Python loop with a single Triton kernel launch.

Each program handles one (m_block, n_block) tile:
- Determines expert from sorted_expert_ids
- Loads activation via sorted_ids indirect indexing
- Loads expert-specific weights
- Computes FP4 GEMM via tl.dot_scaled
- Writes output to (token, slot, n) position

For E=257: uses fused_moe (pre-compiled CK kernels)
For E=33: uses this custom Triton kernel
"""
import torch
import triton
import triton.language as tl
from task import input_t, output_t

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, moe_sorting, get_inter_dim
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort


@triton.jit
def _moe_gemm_kernel(
    # Activation (fp4x2)
    a_ptr, a_scale_ptr,
    # Weight (fp4x2) [E, N, K//2] flattened as [E*N, K//2]
    w_ptr, w_scale_ptr,
    # MoE indexing
    sorted_ids_ptr, sorted_expert_ids_ptr,
    # Output [total_sorted_tokens, N] bf16
    out_ptr,
    # Dims
    num_valid_tokens,  # total valid sorted tokens
    topk, K, N, E,
    # Weight strides (per-expert: w[expert_id, n, k])
    stride_w_e, stride_w_n, stride_w_k,
    # Weight scale: [E*N, K//32] - stride per row
    stride_ws_row, stride_ws_col,
    # Activation strides
    stride_a_m, stride_a_k,
    stride_as_m, stride_as_k,
    # Output strides
    stride_o_m, stride_o_n,
    # Block config
    block_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """MoE GEMM: for each sorted token block, compute activation @ expert_weight.T"""
    SCALE_GROUP: tl.constexpr = 32

    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    # Which expert block is this?
    expert_block = pid_m * BLOCK_M // block_size
    expert_id = tl.load(sorted_expert_ids_ptr + expert_block)

    # Token offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K // 2)

    # Load sorted token IDs and compute original token index
    token_ids = tl.load(sorted_ids_ptr + offs_m)
    m_valid = token_ids < num_valid_tokens
    # Clamp to valid range to prevent OOB memory access (masked out in stores)
    safe_token_ids = tl.where(m_valid, token_ids, 0)
    orig_token = safe_token_ids // topk

    # Activation pointers: a[orig_token, k]
    a_ptrs = a_ptr + orig_token[:, None] * stride_a_m + offs_k[None, :] * stride_a_k

    # Activation scale pointers: as[orig_token, k_scale]
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP)
    as_ptrs = a_scale_ptr + orig_token[:, None] * stride_as_m + offs_ks[None, :] * stride_as_k

    # Weight pointers: w[expert_id, n, k]
    w_base = w_ptr + expert_id * stride_w_e
    w_ptrs = w_base + offs_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n

    # Weight scale pointers: ws[expert_id * N + n, k_scale]
    ws_base = w_scale_ptr + expert_id * N * stride_ws_row
    ws_ptrs = ws_base + offs_n[:, None] * stride_ws_row + offs_ks[None, :] * stride_ws_col

    # Main GEMM loop
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k_iters = tl.cdiv(K, BLOCK_K)

    for ki in range(num_k_iters):
        # Load activations
        a = tl.load(a_ptrs, mask=m_valid[:, None], other=0)
        a_scales = tl.load(as_ptrs, mask=m_valid[:, None], other=0)

        # Load weights
        w = tl.load(w_ptrs)
        w_scales = tl.load(ws_ptrs)

        # FP4 GEMM via native MFMA
        accumulator = tl.dot_scaled(a, a_scales, "e2m1", w, w_scales, "e2m1", accumulator)

        # Advance K pointers
        a_ptrs += (BLOCK_K // 2) * stride_a_k
        w_ptrs += (BLOCK_K // 2) * stride_w_k
        as_ptrs += (BLOCK_K // SCALE_GROUP) * stride_as_k
        ws_ptrs += (BLOCK_K // SCALE_GROUP) * stride_ws_col

    # Store output
    c = accumulator.to(tl.bfloat16)
    out_ptrs = out_ptr + offs_m[:, None] * stride_o_m + offs_n[None, :] * stride_o_n
    out_mask = m_valid[:, None] & (offs_n[None, :] < N)
    tl.store(out_ptrs, c, mask=out_mask)


def moe_gemm(
    a_q, a_scale,  # [M, K//2] uint8, [M, K//32] uint8
    w, w_scale,    # [E, N, K//2] uint8, [E*N, K//32] uint8
    sorted_ids, sorted_expert_ids,
    num_valid_tokens, topk, block_size,
    E, N, K,
):
    """Launch the MoE GEMM kernel."""
    max_sorted = sorted_ids.shape[0]
    device = a_q.device

    # Output buffer
    out = torch.empty((max_sorted, N), dtype=torch.bfloat16, device=device)

    BLOCK_M = block_size  # Match MoE block size
    BLOCK_N = 128
    BLOCK_K = 128

    num_m_blocks = triton.cdiv(max_sorted, BLOCK_M)
    num_n_blocks = triton.cdiv(N, BLOCK_N)
    grid = (num_m_blocks * num_n_blocks,)

    _moe_gemm_kernel[grid](
        a_q, a_scale,
        w.view(torch.uint8), w_scale.view(torch.uint8),
        sorted_ids, sorted_expert_ids,
        out,
        num_valid_tokens, topk, K, N, E,
        w.stride(0), w.stride(1), w.stride(2),
        w_scale.stride(0), w_scale.stride(1),
        a_q.stride(0), a_q.stride(1),
        a_scale.stride(0), a_scale.stride(1),
        out.stride(0), out.stride(1),
        block_size=block_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


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

    # For large E or large d_expert, use fused_moe (pre-compiled CK)
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

    # Custom Triton MoE for E=33
    N1 = gate_up_weight.shape[1]  # 2*d_expert_pad
    N2 = down_weight.shape[1]     # d_hidden_pad
    K1 = gate_up_weight.shape[2] * 2  # d_hidden (unpacked from fp4x2)
    K2 = down_weight.shape[2] * 2     # d_expert_pad (unpacked)
    _, model_dim, inter_dim = get_inter_dim(
        gate_up_weight_shuffled.shape, down_weight_shuffled.shape
    )
    dtype = hidden_states.dtype
    device = hidden_states.device

    block_m = 32

    # Step 1: Sort
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m,
    )
    num_valid = M * topk  # total valid token-slot pairs

    # Step 2: Quantize input
    h_q, h_scale = dynamic_mxfp4_quant(hidden_states)

    # Step 3: Stage 1 MoE GEMM (gate_up)
    gate_up_out = moe_gemm(
        h_q.view(torch.uint8), h_scale.view(torch.uint8),
        gate_up_weight, gate_up_weight_scale,
        sorted_ids, sorted_expert_ids,
        num_valid, topk, block_m,
        E, N1, K1,
    )

    # Step 4: SwiGLU activation (on sorted output)
    # gate_up_out: [max_sorted, N1] where N1 = 2*d_expert_pad
    gate = gate_up_out[:, :d_expert_pad].float()
    up = gate_up_out[:, d_expert_pad:].float()
    intermediate = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)

    # Step 5: Quantize intermediate
    inter_q, inter_s = dynamic_mxfp4_quant(intermediate)

    # Step 6: Stage 2 MoE GEMM (down)
    down_out = moe_gemm(
        inter_q.view(torch.uint8), inter_s.view(torch.uint8),
        down_weight, down_weight_scale,
        sorted_ids, sorted_expert_ids,
        num_valid, topk, block_m,
        E, N2, K2,
    )

    # Step 7: Weighted reduction (vectorized)
    # down_out[sorted_pos, :] -> output[token_idx, :d_hidden] weighted by sorted_weights
    valid_mask = sorted_ids < num_valid
    valid_sids = sorted_ids[valid_mask]
    valid_weights = sorted_weights[valid_mask]
    valid_down = down_out[valid_mask, :d_hidden]

    token_indices = valid_sids // topk
    weighted = valid_weights.unsqueeze(1) * valid_down.float()

    output = torch.zeros((M, d_hidden), dtype=torch.float32, device=device)
    output.index_add_(0, token_indices.long(), weighted)

    return output.to(torch.bfloat16)
