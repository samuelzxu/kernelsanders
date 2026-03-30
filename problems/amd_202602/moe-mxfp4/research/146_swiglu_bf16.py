"""
v146: Try Swiglu + bf16 activations to skip quant for ALL shapes.
Hypothesis: Swiglu path passes BF16 directly to CK kernels, eliminating
quant (25s compile + per-call overhead) and requant steps.
Falls back to v103 if Swiglu fails.
"""
import os, sys, stat

_JIT_DIR = "/home/runner/aiter/aiter/jit"
_BASE_URL = "https://github.com/samuelzxu/aiter-precompiled/releases/download/v0.3-rocm71"
_MODULES = [
    "module_aiter_enum.so",
    "module_moe_sorting_opus.so",
    "module_quant.so",
    "module_activation.so",
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
os.environ['AITER_USE_NT'] = '1'

import torch
import functools
import aiter
from task import input_t, output_t

from aiter import ActivationType, QuantType, dtypes
import aiter.fused_moe as _fm
from aiter.fused_moe import (
    fused_moe_2stages, get_inter_dim,
    get_padded_M, get_2stage_cfgs,
    cktile_moe_stage1, cktile_moe_stage2,
)

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
    _sort_fwd(
        topk_ids, topk_weights,
        sorted_ids, sorted_weights,
        sorted_expert_ids, num_valid_ids,
        moe_buf, E, int(block_size_M),
        None, None, 0,
    )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


_swiglu_failed = False

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _swiglu_failed
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    M, topk = topk_ids.shape
    w1, w2 = gate_up_weight_shuffled, down_weight_shuffled
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    isG1U1 = inter_dim != w1.shape[1]
    dtype = hidden_states.dtype
    is_shuffled = getattr(w1, "is_shuffled", False)
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    tokens_per_expert = (M * topk) / E

    # For dense shapes, try Swiglu + BF16 activations (skips quant)
    use_swiglu = not _swiglu_failed and inter_dim <= 1024 and tokens_per_expert >= 5
    activation = ActivationType.Swiglu if use_swiglu else ActivationType.Silu
    q_dtype_a = dtypes.bf16 if use_swiglu else dtypes.fp4x2

    metadata = _patched(
        get_padded_M(M), model_dim, inter_dim, E, topk,
        dtype, q_dtype_a, dtypes.fp4x2,
        QuantType.per_1x32, isG1U1,
        activation, False,
        hidden_pad, intermediate_pad, is_shuffled,
    )
    block_size_M = int(metadata.block_m)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = _fast_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_size_M,
    )

    try:
        fused_moe_2stages(
            hidden_states, w1, w2,
            topk, sorted_ids, sorted_weights,
            sorted_expert_ids, num_valid_ids,
            moe_buf, isG1U1, block_size_M,
            activation=activation,
            quant_type=QuantType.per_1x32,
            doweight_stage1=False,
            q_dtype_a=q_dtype_a, q_dtype_w=dtypes.fp4x2,
            w1_scale=gate_up_weight_scale_shuffled,
            w2_scale=down_weight_scale_shuffled,
            a1_scale=None, a2_scale=None,
            hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
        )
    except Exception as e:
        if use_swiglu:
            print(f"[v146] Swiglu failed: {e}, falling back", file=sys.stderr)
            _swiglu_failed = True
            # Fallback to standard Silu + fp4x2
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
                w1_scale=gate_up_weight_scale_shuffled,
                w2_scale=down_weight_scale_shuffled,
                a1_scale=None, a2_scale=None,
                hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
            )
        else:
            raise

    return moe_buf
