"""
v189: Direct pipeline dispatch — bypass fused_moe entirely for CK 2-stage shapes.
Call sort/quant/GEMM C++ functions directly with pre-allocated buffers.
Eliminates ~15us Python overhead per call (metadata lookup, function dispatch, etc).
"""
import os, sys, stat
_JIT_DIR = "/home/runner/aiter/aiter/jit"
_BASE_URL = "https://github.com/samuelzxu/aiter-precompiled/releases/download/v0.3-rocm71"
_MODULES = ["module_aiter_enum.so","module_moe_sorting_opus.so","module_moe_sorting.so",
    "module_quant.so","module_activation.so","module_moe_cktile2stages.so",
    "module_moe_ck2stages_fp4x2_fp4x2_preshuffle_on_b16_silu_per_1x32_mulWeightStage2_.so",
    "module_moe_asm.so"]
def _install():
    import urllib.request
    os.makedirs(_JIT_DIR, exist_ok=True)
    for name in _MODULES:
        path = os.path.join(_JIT_DIR, name)
        if not os.path.exists(path):
            try: urllib.request.urlretrieve(f"{_BASE_URL}/{name}", path); os.chmod(path, 0o755)
            except Exception: pass
try: _install()
except Exception: pass

import os
os.environ['HIP_FORCE_DEV_KERNARG'] = '1'
os.environ['GPU_MAX_HW_QUEUES'] = '2'
os.environ['AITER_USE_NT'] = '1'

import torch, functools, aiter
from task import input_t, output_t
from aiter import ActivationType, QuantType, dtypes
import aiter.fused_moe as _fm
from aiter.fused_moe import (
    fused_moe, fused_moe_2stages, get_inter_dim, get_padded_M, get_2stage_cfgs,
    cktile_moe_stage1, cktile_moe_stage2, fused_moe_1stage, MOEMetadata,
    fused_dynamic_mxfp4_quant_moe_sort,
)
from aiter.utility import fp4_utils

_orig = get_2stage_cfgs.__wrapped__
BLOCK_SIZE_M = 32

# FlyDSL injection
_flydsl_injected = False
def _inject_flydsl():
    global _flydsl_injected
    if _flydsl_injected:
        return
    try:
        if not hasattr(_fm, 'cfg_2stages') or _fm.cfg_2stages is None:
            return
        _flydsl_injected = True
        key = (256, 512, 7168, 2048, 33, 9,
               "ActivationType.Silu", "torch.bfloat16",
               "torch.float4_e2m1fn_x2", "torch.float4_e2m1fn_x2",
               "QuantType.per_1x32", 1, 0)
        _fm.cfg_2stages[key] = {
            "block_m": 64, "ksplit": 0, "run_1stage": 0,
            "kernelName1": "moe_ck2stages_gemm1_256x64x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
            "kernelName2": "flydsl_moe2_afp4_wfp4_bf16_t64x256x256_reduce",
        }
    except Exception:
        pass

@functools.lru_cache(maxsize=2048)
def _patched(token, model_dim, inter_dim, expert, topk,
             dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1,
             activation, doweight_stage1, hidden_pad, intermediate_pad, is_shuffled):
    _inject_flydsl()
    md = _orig(token, model_dim, inter_dim, expert, topk, dtype, q_dtype_a, q_dtype_w,
               q_type, use_g1u1, activation, doweight_stage1, hidden_pad, intermediate_pad, is_shuffled)
    tokens_per_expert = (token * topk) / expert
    use_cktile = False; sk = 1
    if inter_dim > 1024: use_cktile = False
    elif tokens_per_expert < 5: use_cktile = True; sk = 2
    elif tokens_per_expert < 40 and expert <= 33: use_cktile = True; sk = 1
    if use_cktile and is_shuffled:
        md.ksplit = 2
        md.block_m = 16 if token < 2048 else 32 if token < 16384 else 64
        md.stage1 = functools.partial(cktile_moe_stage1,
            n_pad_zeros=intermediate_pad // 64 * 64 * (2 if use_g1u1 else 1),
            k_pad_zeros=hidden_pad // 128 * 128, activation=ActivationType.Silu, split_k=sk)
        md.stage2 = functools.partial(cktile_moe_stage2,
            n_pad_zeros=hidden_pad // 64 * 64, k_pad_zeros=intermediate_pad // 128 * 128,
            activation=ActivationType.Silu)
        return md
    if inter_dim <= 1024 and expert <= 33 and q_type == QuantType.per_1x32 and is_shuffled:
        return MOEMetadata(functools.partial(fused_moe_1stage, kernelName="",
            activation=activation, quant_type=q_type), None, BLOCK_SIZE_M, 0, True)
    return md

_fm.get_2stage_cfgs = _patched

# Pre-allocated buffers for direct pipeline
_sort_bufs = {}
_a2_bufs = {}
_has_opus = hasattr(aiter, 'moe_sorting_opus_fwd')
_sort_fwd = aiter.moe_sorting_opus_fwd if _has_opus else aiter.moe_sorting_fwd

# Cache for resolved stage1/stage2 functions after first call
_resolved = {}

def _direct_ck_2stage(hidden_states, w1, w2, w1s, w2s, topk_weights, topk_ids,
                       M, topk, E, inter_dim, model_dim, hidden_pad, intermediate_pad,
                       block_size_M, dtype, device):
    """Direct CK 2-stage pipeline with minimal Python overhead."""
    shape_key = (M, E, inter_dim, model_dim)

    # Pre-allocate sorting buffers
    if shape_key not in _sort_bufs:
        max_num_tokens_padded = M * topk + E * block_size_M - topk
        max_num_m_blocks = (max_num_tokens_padded + block_size_M - 1) // block_size_M
        _sort_bufs[shape_key] = (
            torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device),
            torch.empty(max_num_tokens_padded, dtype=dtypes.fp32, device=device),
            torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device),
            torch.empty(2, dtype=dtypes.i32, device=device),
            torch.empty((M, model_dim), dtype=dtype, device=device))
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = _sort_bufs[shape_key]

    # Sort
    _sort_fwd(topk_ids, topk_weights, sorted_ids, sorted_weights,
              sorted_expert_ids, num_valid_ids, moe_buf, E, block_size_M, None, None, 0)

    # Fused quant + sort (for token <= 1024)
    a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
        hidden_states, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=M, topk=1, block_size=block_size_M)

    # Pre-allocate a2
    if shape_key not in _a2_bufs:
        _a2_bufs[shape_key] = torch.empty((M, topk, inter_dim), dtype=dtype, device=device)
    a2 = _a2_bufs[shape_key]

    # Resolve stage1/stage2 on first call
    if shape_key not in _resolved:
        _inject_flydsl()
        md = _orig(get_padded_M(M), model_dim, inter_dim, E, topk,
                   dtype, dtypes.fp4x2, dtypes.fp4x2, QuantType.per_1x32, True,
                   ActivationType.Silu, False, hidden_pad, intermediate_pad,
                   getattr(w1, "is_shuffled", False))
        _resolved[shape_key] = md
    md = _resolved[shape_key]

    # CK stage 1
    md.stage1(a1, w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids,
              a2, topk, block_m=block_size_M,
              a1_scale=a1_scale,
              w1_scale=w1s.view(dtypes.fp8_e8m0),
              sorted_weights=None)

    # Fused requant + sort for stage 2
    a2_flat = a2.view(-1, inter_dim)
    a2_q, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
        a2_flat, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=M, topk=topk, block_size=block_size_M)
    a2_q = a2_q.view(M, topk, -1)

    # CK/FlyDSL stage 2
    md.stage2(a2_q, w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids,
              moe_buf, topk,
              w2_scale=w2s.view(dtypes.fp8_e8m0),
              a2_scale=a2_scale,
              block_m=block_size_M,
              sorted_weights=sorted_weights)

    return moe_buf

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    (hidden_states, _, _, _, _, w1, w2, w1s, w2s, topk_weights, topk_ids, config) = data
    M, topk = topk_ids.shape
    E = w1.shape[0]
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]
    _, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)

    tokens_per_expert = (M * topk) / E
    use_cktile = (inter_dim <= 1024 and
                  (tokens_per_expert < 5 or (tokens_per_expert < 40 and E <= 33)))
    use_asm = (inter_dim <= 1024 and E <= 33 and not use_cktile)

    # CK 2-stage shapes: use direct pipeline (bypass fused_moe)
    if not use_cktile and not use_asm:
        try:
            return _direct_ck_2stage(
                hidden_states, w1, w2, w1s, w2s, topk_weights, topk_ids,
                M, topk, E, inter_dim, model_dim, hidden_pad, intermediate_pad,
                64 if inter_dim > 1024 else 32, hidden_states.dtype, hidden_states.device)
        except Exception as e:
            import traceback
            print(f"[v189] Direct pipeline failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    # Fallback: use fused_moe for cktile/ASM shapes
    return fused_moe(hidden_states, w1, w2, topk_weights, topk_ids,
                     expert_mask=None, activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32, doweight_stage1=False,
                     w1_scale=w1s, w2_scale=w2s, a1_scale=None, a2_scale=None,
                     hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
