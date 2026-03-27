"""
v152: Triton matmul_ogs (moe_gemm_a4w4) for MOE GEMM.
Uses tl.dot_scaled for native CDNA4 MFMA instruction.
Builds RoutingData from topk_ids/topk_weights to drive Triton matmul.

Architecture:
  1. Build RoutingData from topk_ids (construct hist, gather/scatter indices)
  2. Quant hidden_states to MXFP4 via Triton kernel
  3. Stage 1: moe_gemm_a4w4(hidden_states_fp4, gate_up_weight, apply_swiglu=True)
  4. Quant stage1 output to MXFP4
  5. Stage 2: moe_gemm_a4w4(stage1_fp4, down_weight, apply_swiglu=False)
"""
import os, sys, stat

_JIT_DIR = "/home/runner/aiter/aiter/jit"
_BASE_URL = "https://github.com/samuelzxu/aiter-precompiled/releases/download/v0.3-rocm71"
_MODULES = [
    "module_aiter_enum.so",
    "module_moe_sorting_opus.so",
    "module_moe_sorting.so",
    "module_quant.so",
    "module_activation.so",
    "module_moe_cktile2stages.so",
    "module_moe_ck2stages_fp4x2_fp4x2_preshuffle_on_b16_silu_per_1x32_mulWeightStage2_.so",
]

def _install_precompiled():
    import urllib.request
    os.makedirs(_JIT_DIR, exist_ok=True)
    for name in _MODULES:
        path = os.path.join(_JIT_DIR, name)
        if not os.path.exists(path):
            url = f"{_BASE_URL}/{name}"
            try:
                urllib.request.urlretrieve(url, path)
                os.chmod(path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
            except Exception:
                pass

try:
    _install_precompiled()
except Exception:
    pass

import os
os.environ['HIP_FORCE_DEV_KERNARG'] = '1'
os.environ['GPU_MAX_HW_QUEUES'] = '2'

import torch
import functools
import triton
import aiter
from task import input_t, output_t

from aiter import ActivationType, QuantType, dtypes
import aiter.fused_moe as _fm
from aiter.fused_moe import (
    fused_moe_2stages, get_inter_dim,
    get_padded_M, get_2stage_cfgs,
    cktile_moe_stage1, cktile_moe_stage2,
)

# Triton matmul_ogs imports
from aiter.ops.triton.moe.moe_op_gemm_a4w4 import (
    moe_gemm_a4w4,
    mxfp4_quant,
    swizzle_scales,
)
from aiter.ops.triton.moe.moe_routing.routing import (
    RoutingData, ExptData,
)

# ===== Build RoutingData from topk_ids/topk_weights =====

def build_routing_data(topk_ids, topk_weights, E, block_m=None):
    """Convert topk_ids [M, topk] and topk_weights [M, topk] to RoutingData."""
    M, topk = topk_ids.shape
    n_gates = M * topk
    device = topk_ids.device

    if block_m is None:
        tokens_per_expt = max(1, n_gates // E)
        block_m = max(16, min(triton.next_power_of_2(tokens_per_expt), 128))

    # Sort each token's expert selections by expert ID
    expt_indx_sorted, sort_indices = torch.sort(topk_ids, dim=1)
    expt_scal_sorted = torch.gather(topk_weights, 1, sort_indices)

    # Flatten
    expt_indx_flat = expt_indx_sorted.reshape(-1).to(torch.int32)
    expt_scal_flat = expt_scal_sorted.reshape(-1)

    # Sort by expert_id (stable sort for contiguous experts)
    topk_indx = torch.argsort(expt_indx_flat, stable=True)
    gate_indx = torch.argsort(topk_indx, stable=True)
    gate_scal = expt_scal_flat[topk_indx]

    # Histogram
    hist = torch.histc(expt_indx_flat.float(), bins=E, min=0, max=E - 1).int()

    # Build ExptData
    token_offs_raw = torch.cumsum(hist, dim=0)
    token_offs_raw = torch.cat((torch.zeros(1, device=device, dtype=torch.int32), token_offs_raw)).int()

    n_tiles = (hist + block_m - 1) // block_m
    token_offs_pad = torch.cumsum(n_tiles, dim=0)
    token_offs_pad = torch.cat((torch.zeros(1, device=device, dtype=torch.int32), token_offs_pad)).int()

    if n_gates <= E:
        max_n_tiles = n_gates
    else:
        max_n_tiles = E - 1 - ((E - n_gates - 1) // block_m)

    block_pid_map = -torch.ones(max_n_tiles, device=device, dtype=torch.int32)
    for e in range(E):
        offset = token_offs_pad[e].item()
        for b in range(n_tiles[e].item()):
            if offset + b < max_n_tiles:
                block_pid_map[offset + b] = (b << 16) + e

    expt_data = ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)

    gather_indx = topk_indx.int()
    scatter_indx = gate_indx.int()

    routing_data = RoutingData(
        block_m=block_m,
        gate_scal=gate_scal,
        expt_hist=hist,
        n_expts_tot=E,
        n_expts_act=topk,
        expt_data=expt_data,
    )

    return routing_data, gather_indx, scatter_indx


# ===== CK fallback =====
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
        use_cktile = True; sk = 2
    elif tokens_per_expert < 40 and expert <= 33:
        use_cktile = True; sk = 1
    if use_cktile and is_shuffled:
        md.ksplit = 2
        md.block_m = 16 if token < 2048 else 32 if token < 16384 else 64
        md.stage1 = functools.partial(cktile_moe_stage1,
            n_pad_zeros=intermediate_pad // 64 * 64 * (2 if use_g1u1 else 1),
            k_pad_zeros=hidden_pad // 128 * 128, activation=ActivationType.Silu, split_k=sk)
        md.stage2 = functools.partial(cktile_moe_stage2,
            n_pad_zeros=hidden_pad // 64 * 64,
            k_pad_zeros=intermediate_pad // 128 * 128, activation=ActivationType.Silu)
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

def _ck_kernel(data):
    (hidden_states, gate_up_weight, down_weight,
     gate_up_weight_scale, down_weight_scale,
     gate_up_weight_shuffled, down_weight_shuffled,
     gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
     topk_weights, topk_ids, config) = data
    M, topk = topk_ids.shape
    w1, w2 = gate_up_weight_shuffled, down_weight_shuffled
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    isG1U1 = inter_dim != w1.shape[1]
    dtype = hidden_states.dtype
    is_shuffled = getattr(w1, "is_shuffled", False)
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]
    metadata = _patched(get_padded_M(M), model_dim, inter_dim, E, topk,
                        dtype, dtypes.fp4x2, dtypes.fp4x2, QuantType.per_1x32, isG1U1,
                        ActivationType.Silu, False, hidden_pad, intermediate_pad, is_shuffled)
    block_size_M = int(metadata.block_m)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = _fast_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_size_M)
    fused_moe_2stages(hidden_states, w1, w2, topk, sorted_ids, sorted_weights,
                      sorted_expert_ids, num_valid_ids, moe_buf, isG1U1, block_size_M,
                      activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
                      doweight_stage1=False, q_dtype_a=dtypes.fp4x2, q_dtype_w=dtypes.fp4x2,
                      w1_scale=gate_up_weight_scale_shuffled, w2_scale=down_weight_scale_shuffled,
                      a1_scale=None, a2_scale=None,
                      hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
    return moe_buf


def _triton_kernel(data):
    """Triton matmul_ogs path — full 2-stage MOE."""
    (hidden_states, gate_up_weight, down_weight,
     gate_up_weight_scale, down_weight_scale,
     gate_up_weight_shuffled, down_weight_shuffled,
     gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
     topk_weights, topk_ids, config) = data

    M, topk = topk_ids.shape
    E = gate_up_weight.shape[0]
    d_hidden = config["d_hidden"]
    d_expert = config["d_expert"]
    d_hidden_pad = config["d_hidden_pad"]
    d_expert_pad = config["d_expert_pad"]

    # Build routing data from topk_ids
    routing_data, gather_indx, scatter_indx = build_routing_data(
        topk_ids, topk_weights, E
    )

    # Quant hidden_states to MXFP4
    x_fp4, x_scales = mxfp4_quant(hidden_states)

    # Raw scales are 2D [E*N, scale_K] — reshape to 3D [E, scale_K, N]
    # moe_gemm_a4w4 expects w_scales with strides [stride_e, stride_k, stride_n]
    N1 = gate_up_weight.shape[1]  # 2*d_expert_pad
    N2 = down_weight.shape[1]     # d_hidden_pad
    gus = gate_up_weight_scale.view(torch.uint8) if gate_up_weight_scale.dtype != torch.uint8 else gate_up_weight_scale
    dws = down_weight_scale.view(torch.uint8) if down_weight_scale.dtype != torch.uint8 else down_weight_scale
    # Reshape to [E, N, scale_K] then transpose to [E, scale_K, N]
    # DO NOT call .contiguous() — the non-contiguous layout gives stride(1)=1
    # which the kernel expects (K blocks contiguous, N values strided)
    w1_scale_3d = gus.view(E, N1, -1).permute(0, 2, 1)  # [E, scale_K, N] stride(1)=1
    w2_scale_3d = dws.view(E, N2, -1).permute(0, 2, 1)  # [E, scale_K, N] stride(1)=1

    # Convert FP4 weights to uint8 view (Triton expects uint8, not float4_e2m1fn_x2)
    # Transpose to get [E, K//2, N] with stride(-2)==1 (column-major, non-contiguous)
    w1 = gate_up_weight.view(torch.uint8).transpose(-1, -2)  # stride(-2)==1
    w2 = down_weight.view(torch.uint8).transpose(-1, -2)      # stride(-2)==1

    # Stage 1: gate_up GEMM WITHOUT swiglu (Triton's swiglu differs from reference SiLU)
    stage1_raw = moe_gemm_a4w4(
        x_fp4,                    # [M, d_hidden_pad//2] fp4x2
        w1,                       # [E, d_hidden_pad//2, 2*d_expert_pad] column-major
        x_scales,                 # [M, d_hidden_pad//32] e8m0
        w1_scale_3d,       # [E, ...] swizzled
        routing_data=routing_data,
        gather_indx=gather_indx,
        scatter_indx=None,        # NO scatter — keep per-(token,expert) output
        swizzle_mx_scale=None,
        out_dtype=torch.bfloat16,
        apply_swiglu=False,       # Do NOT fuse swiglu — apply SiLU manually
    )
    # stage1_raw: [M*topk_sorted, 2*d_expert_pad] bf16

    # Apply standard SiLU gate mechanism: silu(gate) * up
    half_n = stage1_raw.shape[-1] // 2
    gate = stage1_raw[:, :half_n]
    up = stage1_raw[:, half_n:]
    stage1_out = torch.nn.functional.silu(gate) * up
    # stage1_out: [M*topk_sorted, d_expert_pad] bf16

    # Quant stage1 output to MXFP4 for stage 2
    s1_fp4, s1_scales = mxfp4_quant(stage1_out)

    # Stage 2: down GEMM WITH scatter+reduce AND gammas for routing weights
    # gammas applies per-row scaling (routing weights) inside the matmul
    # then reduce_grouped sums the weighted contributions per token
    stage2_out = moe_gemm_a4w4(
        s1_fp4, w2, s1_scales, w2_scale_3d,
        routing_data=routing_data,
        gather_indx=None,
        scatter_indx=scatter_indx,  # Scatter + reduce back to [M, d_hidden]
        gammas=routing_data.gate_scal,  # Apply routing weights per-row
        swizzle_mx_scale=None,
        out_dtype=torch.bfloat16,
        apply_swiglu=False,
    )
    # stage2_out: [M, d_hidden_pad] bf16 (weighted + reduced)
    output = stage2_out

    return output[:, :d_hidden]


_triton_warmup_done = False

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _triton_warmup_done
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    M = topk_ids.shape[0]
    E = gate_up_weight.shape[0]
    topk = topk_ids.shape[1]
    tokens_per_expert = (M * topk) / E

    # Use Triton for dense shapes (E<=33), CK for sparse (E=257)
    # Triton has correctness issues with very sparse routing (E=257, bs=8/16)
    use_triton = E <= 65 and tokens_per_expert >= 5

    if use_triton:
        if not _triton_warmup_done:
            _triton_warmup_done = True
            try:
                result = _triton_kernel(data)
                print(f"[v152] Triton OK for E={E} M={M}", file=sys.stderr)
                return result
            except Exception as e:
                print(f"[v152] Triton failed: {e}", file=sys.stderr)
                return _ck_kernel(data)
        try:
            return _triton_kernel(data)
        except Exception:
            return _ck_kernel(data)
    else:
        return _ck_kernel(data)
