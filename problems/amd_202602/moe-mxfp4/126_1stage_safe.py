"""
v126: Use 1-stage fused kernel for shapes where it's available and correct.
The CK 1-stage kernel (aiter.fmoe_g1u1) fuses both GEMMs into one launch,
eliminating: a2 buffer allocation, requant step, and 2nd kernel launch.
Only use for d_expert <= 1536 (d=2048 has precision issues from v33/v63).
Falls back to 2-stage if ASM module isn't pre-compiled on runner.
"""
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
    fused_moe_2stages, fused_moe_1stage, get_inter_dim,
    get_padded_M, get_2stage_cfgs,
    cktile_moe_stage1, cktile_moe_stage2,
)

# Check if ASM 1-stage module is pre-compiled (avoid triggering JIT build)
_asm_available = False
try:
    _asm_path = os.path.join(
        os.path.dirname(aiter.__file__), 'jit', 'module_moe_asm.so'
    )
    if os.path.exists(_asm_path):
        # Module exists, try to load it (should be fast if pre-compiled)
        _asm_available = hasattr(aiter, 'fmoe_g1u1') or True  # will be loaded on first use
except Exception:
    _asm_available = False

_orig = get_2stage_cfgs.__wrapped__
_silu = ActivationType.Silu
_per_1x32 = QuantType.per_1x32
_fp4x2 = dtypes.fp4x2
_i32 = dtypes.i32
_fp32 = dtypes.fp32

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
            activation=_silu,
            split_k=sk,
        )
        md.stage2 = functools.partial(
            cktile_moe_stage2,
            n_pad_zeros=hidden_pad // 64 * 64,
            k_pad_zeros=intermediate_pad // 128 * 128,
            activation=_silu,
        )

    return md

_fm.get_2stage_cfgs = _patched

_sorting_bufs = {}
_has_opus = hasattr(aiter, 'moe_sorting_opus_fwd')
_sort_fwd = aiter.moe_sorting_opus_fwd if _has_opus else aiter.moe_sorting_fwd

# Track whether 1-stage actually works (verified on first call)
_1stage_verified = {}


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _asm_available

    (
        hidden_states, _, _,
        _, _,
        w1, w2,
        w1_scale, w2_scale,
        topk_weights, topk_ids, config,
    ) = data

    M = hidden_states.shape[0]
    topk = topk_ids.shape[1]
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    isG1U1 = inter_dim != w1.shape[1]
    dtype = hidden_states.dtype
    is_shuffled = getattr(w1, "is_shuffled", False)
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    metadata = _patched(
        get_padded_M(M), model_dim, inter_dim, E, topk,
        dtype, _fp4x2, _fp4x2,
        _per_1x32, isG1U1,
        _silu, False,
        hidden_pad, intermediate_pad, is_shuffled,
    )
    block_size_M = int(metadata.block_m)

    # Pre-allocated sorting
    sort_key = (M, E, model_dim, block_size_M)
    if sort_key not in _sorting_bufs:
        device = topk_ids.device
        max_num_tokens_padded = M * topk + E * block_size_M - topk
        max_num_m_blocks = (max_num_tokens_padded + block_size_M - 1) // block_size_M
        _sorting_bufs[sort_key] = (
            torch.empty(max_num_tokens_padded, dtype=_i32, device=device),
            torch.empty(max_num_tokens_padded, dtype=_fp32, device=device),
            torch.empty(max_num_m_blocks, dtype=_i32, device=device),
            torch.empty(2, dtype=_i32, device=device),
            torch.empty((M, model_dim), dtype=dtype, device=device),
        )

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = _sorting_bufs[sort_key]

    _sort_fwd(
        topk_ids, topk_weights,
        sorted_ids, sorted_weights,
        sorted_expert_ids, num_valid_ids,
        moe_buf, E, block_size_M,
        None, None, 0,
    )

    # Use 1-stage for shapes where it works (d_expert <= 1536, ASM available)
    # 1-stage fuses: quant + stage1 GEMM + activation + stage2 GEMM in one path
    use_1stage = (
        _asm_available
        and inter_dim <= 1536
        and isG1U1
        and metadata.ksplit <= 1  # Not cktile shapes (they have their own optimization)
    )

    if use_1stage:
        try:
            fused_moe_1stage(
                hidden_states, w1, w2,
                topk, sorted_ids, sorted_weights,
                sorted_expert_ids, num_valid_ids,
                moe_buf, isG1U1, block_size_M,
                activation=_silu,
                quant_type=_per_1x32,
                kernelName='',
                q_dtype_a=_fp4x2, q_dtype_w=_fp4x2,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=None, a2_scale=None,
                num_local_tokens=None,
                M=M, device=topk_ids.device,
                doweight_stage1=False,
            )
        except Exception:
            # ASM module failed to load, disable 1-stage
            _asm_available = False
            fused_moe_2stages(
                hidden_states, w1, w2,
                topk, sorted_ids, sorted_weights,
                sorted_expert_ids, num_valid_ids,
                moe_buf, isG1U1, block_size_M,
                activation=_silu,
                quant_type=_per_1x32,
                doweight_stage1=False,
                q_dtype_a=_fp4x2, q_dtype_w=_fp4x2,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=None, a2_scale=None,
                hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
            )
    else:
        fused_moe_2stages(
            hidden_states, w1, w2,
            topk, sorted_ids, sorted_weights,
            sorted_expert_ids, num_valid_ids,
            moe_buf, isG1U1, block_size_M,
            activation=_silu,
            quant_type=_per_1x32,
            doweight_stage1=False,
            q_dtype_a=_fp4x2, q_dtype_w=_fp4x2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=None, a2_scale=None,
            hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
        )

    return moe_buf
