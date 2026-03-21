"""
v140: Pure Triton matmul_ogs MOE — NO CK, saves 216s compilation.
Uses moe_gemm_a4w4 with tl.dot_scaled hitting native CDNA4
v_mfma_scale_f32_16x16x128_f8f6f4 instruction.
"""
import torch
from task import input_t, output_t

import aiter.ops.triton.moe.moe_op_gemm_a4w4 as _a4w4_mod
from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_a4w4
from aiter.ops.triton.moe.moe_routing.routing import RoutingData, ExptData
from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant

# Override get_kernel_config to use FIXED config → only 2 Triton compiles
# (one for swiglu=True, one for swiglu=False)
_FIXED_CONFIG = {
    "block_m": 32,
    "block_n": 128,
    "block_k": 256,
    "num_warps": 4,
    "num_stages": 2,
    "group_m": 4,
    "xcd_swizzle": 8,
    "split_k": 1,
    "w_cache_modifier": ".cg",
    "waves_per_eu": 2,
    "matrix_instr_nonkdim": 16,
    "kpack": 1,
}
_a4w4_mod.get_kernel_config = lambda m, n, k, rd: _FIXED_CONFIG

# Weight transpose cache — weights are fixed, transpose once on first call
_w_cache = {}


def _get_transposed_weights(w_fp4, w_scale):
    """Transpose [E, N, K//2] → [E, K//2, N] column-major. Cached."""
    key = w_fp4.data_ptr()
    if key not in _w_cache:
        # w_fp4 is [E, N, K//2] row-major. Kernel needs [E, K//2, N] column-major
        # (stride(-2)==1). This means K//2 dim is contiguous (Fortran order for last 2 dims).
        # Simplest: just use the original tensor as [E, N, K//2] and tell kernel
        # it's [E, K//2, N] column-major — because [E, N, K//2] row-major IS
        # [E, K//2, N] column-major when you read the strides!
        # Row-major [E, N, K//2]: stride = (N*K//2, K//2, 1)
        # If we VIEW as [E, K//2, N]: stride = (N*K//2, 1, K//2)
        # stride(-2) = 1 ✓ column-major!
        E, N, Kh = w_fp4.shape
        wt = w_fp4.reshape(E, N * Kh).reshape(E, Kh, N)  # doesn't work, need real stride

        # Actually: use torch.as_strided to create a view with column-major strides
        # Original [E, N, K//2] with strides (N*Kh, Kh, 1)
        # View as [E, K//2, N] with strides (N*Kh, 1, Kh) — column-major!
        wt = torch.as_strided(w_fp4, (E, Kh, N), (N * Kh, 1, Kh))

        # Scale handling
        if w_scale.ndim == 3:
            Es, Ns, Sk = w_scale.shape
            st = torch.as_strided(w_scale, (Es, Sk, Ns), (Ns * Sk, 1, Sk))
        elif w_scale.ndim == 2:
            E2 = w_fp4.shape[0]
            N2 = w_fp4.shape[1]
            Sk = w_scale.shape[-1] // N2 if w_scale.shape[0] == E2 else w_scale.shape[-1]
            sc3 = w_scale.view(E2, N2, -1)
            Sk2 = sc3.shape[2]
            st = torch.as_strided(sc3, (E2, Sk2, N2), (N2 * Sk2, 1, Sk2))
        else:
            st = w_scale
        _w_cache[key] = (wt, st)
    return _w_cache[key]


def _build_routing_data(topk_ids, topk_weights, n_experts, block_m):
    """Build RoutingData from pre-computed topk results."""
    M, topk = topk_ids.shape
    device = topk_ids.device
    n_gates = M * topk

    flat_ids = topk_ids.reshape(-1)
    flat_weights = topk_weights.reshape(-1)

    # Per-expert histogram
    hist = torch.zeros(n_experts, dtype=torch.int32, device=device)
    ones = torch.ones(n_gates, dtype=torch.int32, device=device)
    hist.scatter_add_(0, flat_ids.long(), ones)

    # Token offsets (exclusive cumsum)
    token_offs_raw = torch.zeros(n_experts, dtype=torch.int32, device=device)
    if n_experts > 1:
        torch.cumsum(hist[:-1], dim=0, out=token_offs_raw[1:])

    # Padded offsets (round up to block_m)
    hist_padded = ((hist + block_m - 1) // block_m) * block_m
    token_offs_pad = torch.zeros(n_experts + 1, dtype=torch.int32, device=device)
    torch.cumsum(hist_padded, dim=0, out=token_offs_pad[1:])

    # Sort tokens by expert using argsort (GPU)
    sorted_order = torch.argsort(flat_ids.int()).int()
    gather_indx = sorted_order // topk  # original token index
    gate_scal = flat_weights[sorted_order.long()]
    scatter_indx = sorted_order

    # block_pid_map: maps thread blocks → (expert, block_within_expert)
    total_padded = int(token_offs_pad[-1].item())
    total_blocks = total_padded // block_m
    block_pid_map = torch.full((total_blocks,), -1, dtype=torch.int32, device=device)

    # Fill on CPU (loop over experts — small, fast)
    h = hist.cpu()
    hp = hist_padded.cpu()
    bpm = block_pid_map.cpu()
    offset = 0
    for eid in range(n_experts):
        nt = int(h[eid])
        nb = (nt + block_m - 1) // block_m
        base = offset // block_m
        for bid in range(nb):
            bpm[base + bid] = (bid << 16) | eid
        offset += int(hp[eid])
    block_pid_map.copy_(bpm)

    expt_data = ExptData(
        hist=hist,
        token_offs_raw=token_offs_raw,
        token_offs_pad=token_offs_pad,
        block_pid_map=block_pid_map,
    )

    return RoutingData(
        block_m=block_m,
        gate_scal=gate_scal,
        expt_hist=hist,
        n_expts_tot=n_experts,
        n_expts_act=topk,
        expt_data=expt_data,
    ), gather_indx, scatter_indx


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
    block_m = 32

    # Transpose weights once (cached after first call)
    w1, w1_scales = _get_transposed_weights(gate_up_weight, gate_up_weight_scale)
    w2, w2_scales = _get_transposed_weights(down_weight, down_weight_scale)

    # Build routing from topk data
    routing_data, gather_indx, scatter_indx = _build_routing_data(
        topk_ids, topk_weights, E, block_m
    )

    # Quantize activations to MXFP4 (kernel needs e2m1 format for tl.dot_scaled)
    x_fp4, x_scales = dynamic_mxfp4_quant(hidden_states)

    # Stage 1: gate_up projection + SiLU fused in single Triton kernel
    stage1_out = moe_gemm_a4w4(
        x_fp4, w1, x_scales, w1_scales,
        routing_data=routing_data,
        gather_indx=gather_indx,
        scatter_indx=scatter_indx,
        apply_swiglu=True,
        out_dtype=torch.bfloat16,
    )

    # Requant intermediate to MXFP4 for stage 2
    inter_fp4, inter_scales = dynamic_mxfp4_quant(stage1_out)

    # Stage 2: down projection
    result = moe_gemm_a4w4(
        inter_fp4, w2, inter_scales, w2_scales,
        routing_data=routing_data,
        gather_indx=None,
        scatter_indx=scatter_indx,
        apply_swiglu=False,
        out_dtype=torch.bfloat16,
    )

    return result
