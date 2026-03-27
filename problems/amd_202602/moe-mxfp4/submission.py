"""
v185: Selective ASM + FlyDSL stage 2 for d>1024 shapes.
- cktile for sparse shapes (BF16, no quant)
- ASM 1-stage for E<=33 d<=512 (fused gate+up+down, no quant)
- CK stage 1 + FlyDSL stage 2 for d>1024 (tuned MFMA assembly)
- CK 2-stage with tuned CSV for E=257 (DSv3 config)
Pre-compiled all 8 modules.
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
    fused_moe_2stages, fused_moe_1stage, get_inter_dim, get_padded_M, get_2stage_cfgs,
    cktile_moe_stage1, cktile_moe_stage2, MOEMetadata, get_block_size_M,
)

_orig = get_2stage_cfgs.__wrapped__
BLOCK_SIZE_M = 32

# Inject FlyDSL stage 2 configs for shapes not in the tuned CSV
_flydsl_injected = False
def _inject_flydsl():
    global _flydsl_injected
    if _flydsl_injected:
        return
    try:
        if not hasattr(_fm, 'cfg_2stages') or _fm.cfg_2stages is None:
            return  # Not loaded yet, will retry on next call
        _flydsl_injected = True  # Only mark done AFTER successful injection
        # E=33, d=2048, token=512: use FlyDSL stage 2 with CK stage 1
        # Note: token value must match get_padded_M(M) in the lookup
        key = (256, 512, 7168, 2048, 33, 9,
               "ActivationType.Silu", "torch.bfloat16",
               "torch.float4_e2m1fn_x2", "torch.float4_e2m1fn_x2",
               "QuantType.per_1x32", 1, 0)
        _fm.cfg_2stages[key] = {
            "block_m": 64, "ksplit": 0, "run_1stage": 0,
            "kernelName1": "moe_ck2stages_gemm1_256x64x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
            "kernelName2": "flydsl_moe2_afp4_wfp4_bf16_t64x256x256_atomic",
        }
        # Note: FlyDSL stage 2 is worse for E=257 d=256 shapes (250 vs 246us)
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

    # For sparse shapes (cktile path): keep cktile (already optimal)
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

    # ASM 1-stage ONLY for small expert count (E<=33) with inter_dim<=1024
    # For E=257+, CK 2-stage is faster (efficient GEMM tiles for many experts)
    if inter_dim <= 1024 and expert <= 33 and q_type == QuantType.per_1x32 and is_shuffled:
        return MOEMetadata(
            functools.partial(fused_moe_1stage,
                kernelName="",  # auto-select
                activation=activation, quant_type=q_type),
            None, BLOCK_SIZE_M, 0, True)

    # Default CK 2-stage (with FlyDSL stage 2 if injected via cfg_2stages)
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
            torch.empty((M, model_dim), dtype=dtype, device=device))
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = _sorting_bufs[key]
    _sort_fwd(topk_ids, topk_weights, sorted_ids, sorted_weights,
              sorted_expert_ids, num_valid_ids, moe_buf, E, int(block_size_M), None, None, 0)
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    (hidden_states, _, _, _, _, w1, w2, w1s, w2s, topk_weights, topk_ids, config) = data
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]
    # Use high-level fused_moe which correctly dispatches 1-stage vs 2-stage
    from aiter.fused_moe import fused_moe
    output = fused_moe(hidden_states, w1, w2, topk_weights, topk_ids,
                       expert_mask=None, activation=ActivationType.Silu,
                       quant_type=QuantType.per_1x32, doweight_stage1=False,
                       w1_scale=w1s, w2_scale=w2s, a1_scale=None, a2_scale=None,
                       hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
    return output
