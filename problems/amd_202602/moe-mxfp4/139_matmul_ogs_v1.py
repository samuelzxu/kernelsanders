"""
v139: Use Triton matmul_ogs (moe_gemm_a4w4) for MOE GEMM.
This is a fundamentally different approach from CK 2-stage:
- Fuses routing + GEMM + SiLU in single Triton kernel
- Uses tl.dot_scaled for native MXFP4 hardware acceleration
- XCD swizzling and aggressive tiling (block_n=512, block_k=256)

First attempt - just get it working, optimize later.
Falls back to v103 CK path if matmul_ogs fails.
"""
import os
os.environ['HIP_FORCE_DEV_KERNARG'] = '1'
os.environ['GPU_MAX_HW_QUEUES'] = '2'

import sys
import torch
import functools
import triton
import aiter
from task import input_t, output_t
from dataclasses import dataclass, field

from aiter import ActivationType, QuantType, dtypes
import aiter.fused_moe as _fm
from aiter.fused_moe import (
    fused_moe_2stages, get_inter_dim,
    get_padded_M, get_2stage_cfgs,
    cktile_moe_stage1, cktile_moe_stage2,
)

# Import matmul_ogs MOE GEMM
try:
    from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_a4w4
    from aiter.ops.triton.moe.moe_routing.routing import RoutingData, ExptData
    _ogs_available = True
except ImportError as e:
    print(f"[v139] matmul_ogs import failed: {e}", file=sys.stderr)
    _ogs_available = False


def _build_routing_data(topk_ids, topk_weights, n_experts, block_m):
    """Build RoutingData from pre-computed topk_ids and topk_weights.

    This constructs the ExptData (histogram, offsets, block-pid map)
    needed by moe_gemm_a4w4, without going through the Bitmatrix path.
    """
    M, topk = topk_ids.shape
    device = topk_ids.device
    n_gates = M * topk

    # Flatten topk data
    flat_ids = topk_ids.view(-1)  # [M*topk]
    flat_weights = topk_weights.view(-1)  # [M*topk]

    # Compute per-expert histogram
    hist = torch.zeros(n_experts, dtype=torch.int32, device=device)
    hist.scatter_add_(0, flat_ids.to(torch.int64), torch.ones_like(flat_ids, dtype=torch.int32))

    # Compute token offsets (cumulative sum of histogram)
    token_offs_raw = torch.zeros(n_experts + 1, dtype=torch.int32, device=device)
    torch.cumsum(hist, dim=0, out=token_offs_raw[1:])

    # Compute padded offsets (round up to block_m)
    hist_padded = ((hist + block_m - 1) // block_m) * block_m
    token_offs_pad = torch.zeros(n_experts + 1, dtype=torch.int32, device=device)
    torch.cumsum(hist_padded, dim=0, out=token_offs_pad[1:])

    # Sort tokens by expert: create gather indices
    # For each token-expert pair, find its position in the sorted order
    expert_counts = torch.zeros(n_experts, dtype=torch.int32, device=device)
    gather_indx = torch.empty(n_gates, dtype=torch.int32, device=device)
    gate_scal = torch.empty(n_gates, dtype=topk_weights.dtype, device=device)

    # Simple CPU-side sorting (TODO: GPU sort for larger batches)
    flat_ids_cpu = flat_ids.cpu()
    flat_weights_cpu = flat_weights.cpu()
    token_offs_cpu = token_offs_raw.cpu()
    counts = torch.zeros(n_experts, dtype=torch.int32)
    gather_cpu = torch.empty(n_gates, dtype=torch.int32)
    gate_scal_cpu = torch.empty(n_gates, dtype=topk_weights.dtype)

    for i in range(n_gates):
        eid = int(flat_ids_cpu[i])
        pos = int(token_offs_cpu[eid]) + int(counts[eid])
        counts[eid] += 1
        # gather_indx[pos] = original token index (i // topk)
        gather_cpu[pos] = i // topk
        gate_scal_cpu[pos] = flat_weights_cpu[i]

    gather_indx = gather_cpu.to(device)
    gate_scal = gate_scal_cpu.to(device)

    # Build block_pid_map: maps each block to (expert_id, block_within_expert)
    total_blocks = int(token_offs_pad[-1].item()) // block_m
    block_pid_map = torch.full((total_blocks,), -1, dtype=torch.int32, device=device)

    offset = 0
    for eid in range(n_experts):
        n_tokens_expert = int(hist[eid].item())
        n_blocks_expert = (n_tokens_expert + block_m - 1) // block_m
        for bid in range(n_blocks_expert):
            block_idx = offset // block_m + bid
            if block_idx < total_blocks:
                # Pack (expert_id, block_id) as int32: low 16 = expert, high 16 = block
                block_pid_map[block_idx] = (bid << 16) | eid
        offset += int(hist_padded[eid].item())

    expt_data = ExptData(
        hist=hist,
        token_offs_raw=token_offs_raw[:n_experts],
        token_offs_pad=token_offs_pad,
        block_pid_map=block_pid_map,
    )

    routing_data = RoutingData(
        block_m=block_m,
        gate_scal=gate_scal,
        expt_hist=hist,
        n_expts_tot=n_experts,
        n_expts_act=topk,
        expt_data=expt_data,
    )

    # scatter_indx: maps sorted positions back to original [M, topk] layout
    scatter_indx = torch.empty(n_gates, dtype=torch.int32, device=device)
    scatter_cpu = torch.empty(n_gates, dtype=torch.int32)
    counts.zero_()
    for i in range(n_gates):
        eid = int(flat_ids_cpu[i])
        pos = int(token_offs_cpu[eid]) + int(counts[eid])
        counts[eid] += 1
        scatter_cpu[pos] = i  # maps sorted pos → original flat index
    scatter_indx = scatter_cpu.to(device)

    return routing_data, gather_indx, scatter_indx


# v103 fallback setup
_orig = get_2stage_cfgs.__wrapped__

@functools.lru_cache(maxsize=2048)
def _patched(token, model_dim, inter_dim, expert, topk,
             dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1,
             activation, doweight_stage1,
             hidden_pad, intermediate_pad, is_shuffled):
    md = _orig(token, model_dim, inter_dim, expert, topk,
               dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1,
               activation, doweight_stage1,
               hidden_pad, intermediate_pad, is_shuffled)
    tokens_per_expert = (token * topk) / expert
    use_cktile = False
    sk = 1
    if inter_dim > 1024:
        use_cktile = False
    elif tokens_per_expert < 5:
        use_cktile = True
        sk = 2
    elif tokens_per_expert < 40 and expert <= 33:
        use_cktile = True
        sk = 1
    if use_cktile and is_shuffled:
        md.ksplit = 2
        md.block_m = 16 if token < 2048 else 32 if token < 16384 else 64
        md.stage1 = functools.partial(
            cktile_moe_stage1,
            n_pad_zeros=intermediate_pad // 64 * 64 * (2 if use_g1u1 else 1),
            k_pad_zeros=hidden_pad // 128 * 128,
            activation=ActivationType.Silu,
            split_k=sk,
        )
        md.stage2 = functools.partial(
            cktile_moe_stage2,
            n_pad_zeros=hidden_pad // 64 * 64,
            k_pad_zeros=intermediate_pad // 128 * 128,
            activation=ActivationType.Silu,
        )
    return md

_fm.get_2stage_cfgs = _patched

_sorting_bufs = {}
_has_opus = hasattr(aiter, 'moe_sorting_opus_fwd')
_sort_fwd = aiter.moe_sorting_opus_fwd if _has_opus else aiter.moe_sorting_fwd

def _fast_sorting(topk_ids, topk_weights, E, model_dim, dtype, block_size_M):
    M, topk = topk_ids.shape
    key = (M, E, model_dim, block_size_M)
    if key not in _sorting_bufs:
        device = topk_ids.device
        max_num_tokens_padded = M * topk + E * block_size_M - topk
        max_num_m_blocks = (max_num_tokens_padded + block_size_M - 1) // block_size_M
        _sorting_bufs[key] = (
            torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device),
            torch.empty(max_num_tokens_padded, dtype=dtypes.fp32, device=device),
            torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device),
            torch.empty(2, dtype=dtypes.i32, device=device),
            torch.empty((M, model_dim), dtype=dtype, device=device),
        )
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = _sorting_bufs[key]
    _sort_fwd(topk_ids, topk_weights, sorted_ids, sorted_weights,
              sorted_expert_ids, num_valid_ids, moe_buf, E, int(block_size_M), None, None, 0)
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


def _ck_fallback(hidden_states, w1, w2, topk, topk_ids, topk_weights,
                 w1_scale, w2_scale, config, E, model_dim, inter_dim):
    """CK 2-stage fallback (v103 logic)."""
    isG1U1 = inter_dim != w1.shape[1]
    M = hidden_states.shape[0]
    dtype = hidden_states.dtype
    is_shuffled = getattr(w1, "is_shuffled", False)
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]
    metadata = _patched(
        get_padded_M(M), model_dim, inter_dim, E, topk,
        dtype, dtypes.fp4x2, dtypes.fp4x2,
        QuantType.per_1x32, isG1U1,
        ActivationType.Silu, False,
        hidden_pad, intermediate_pad, is_shuffled,
    )
    block_size_M = int(metadata.block_m)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = _fast_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_size_M,
    )
    fused_moe_2stages(
        hidden_states, w1, w2,
        topk, sorted_ids, sorted_weights,
        sorted_expert_ids, num_valid_ids,
        moe_buf, isG1U1, block_size_M,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        q_dtype_a=dtypes.fp4x2, q_dtype_w=dtypes.fp4x2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=None, a2_scale=None,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )
    return moe_buf


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    M, topk = topk_ids.shape
    E = gate_up_weight.shape[0]
    d_hidden = config["d_hidden"]
    d_expert = config["d_expert"]

    # Use CK fallback for now - matmul_ogs integration WIP
    # TODO: Switch to matmul_ogs path once format conversion is verified
    w1, w2 = gate_up_weight_shuffled, down_weight_shuffled
    _, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)

    if not _ogs_available:
        return _ck_fallback(hidden_states, w1, w2, topk, topk_ids, topk_weights,
                           gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
                           config, E, model_dim, inter_dim)

    # Try matmul_ogs path
    try:
        block_m = 32  # Standard for matmul_ogs

        # Build routing data from topk
        routing_data, gather_indx, scatter_indx = _build_routing_data(
            topk_ids, topk_weights, E, block_m
        )

        # Quant hidden_states to MXFP4 (use AITER's quant)
        from aiter.ops.triton.quant.fused_mxfp4_quant import dynamic_mxfp4_quant
        x_fp4, x_scales = dynamic_mxfp4_quant(hidden_states)

        # Weight format: moe_gemm_a4w4 expects w as [E, K//2, N] column-major
        # Our raw gate_up_weight is [E, 2*d_expert_pad, d_hidden_pad//2] = [E, N, K//2]
        # We need to transpose the last two dims
        w1_raw = gate_up_weight.transpose(-1, -2).contiguous()
        w1_scales_raw = gate_up_weight_scale  # [E, N, scale_K] - check format

        # Stage 1: gate_up projection + SiLU
        stage1_out = moe_gemm_a4w4(
            x_fp4, w1_raw, x_scales, w1_scales_raw,
            routing_data=routing_data,
            gather_indx=gather_indx,
            scatter_indx=scatter_indx,
            apply_swiglu=True,  # Fuse SiLU+gate
            out_dtype=torch.bfloat16,
        )

        # Requant stage1 output for stage 2
        inter_fp4, inter_scales = dynamic_mxfp4_quant(stage1_out)

        # Stage 2: down projection
        w2_raw = down_weight.transpose(-1, -2).contiguous()
        w2_scales_raw = down_weight_scale

        # Need new routing data for stage 2 (same routing, different gather/scatter)
        result = moe_gemm_a4w4(
            inter_fp4, w2_raw, inter_scales, w2_scales_raw,
            routing_data=routing_data,
            gather_indx=None,  # Already gathered from stage 1
            scatter_indx=scatter_indx,
            apply_swiglu=False,
            out_dtype=torch.bfloat16,
        )

        return result

    except Exception as e:
        print(f"[v139] matmul_ogs failed: {e}, falling back to CK", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return _ck_fallback(hidden_states, w1, w2, topk, topk_ids, topk_weights,
                           gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
                           config, E, model_dim, inter_dim)
