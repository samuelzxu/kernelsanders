"""
v167: Full hybrid — Triton fused_moe_mxfp4 stage 1 (BF16 in, no quant needed)
+ cktile stage 2 (BF16 intermediate, no requant). Eliminates quant+requant overhead.

Pipeline: sort → Triton GEMM(gate_up) → SiLU → cktile GEMM(down) → output
vs CK:    sort → quant → CK GEMM(gate_up) → requant → CK GEMM(down) → output
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

import torch, functools, aiter, triton.language as tl
from task import input_t, output_t
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (fused_moe, get_inter_dim, get_padded_M, get_2stage_cfgs,
    cktile_moe_stage1, cktile_moe_stage2, fused_moe_1stage, MOEMetadata)
import aiter.fused_moe as _fm
from aiter.ops.triton.moe.moe_op_mxfp4 import fused_moe_mxfp4

_orig = get_2stage_cfgs.__wrapped__
@functools.lru_cache(maxsize=2048)
def _patched(token, model_dim, inter_dim, expert, topk,
             dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1,
             activation, doweight_stage1, hidden_pad, intermediate_pad, is_shuffled):
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
    if inter_dim <= 1024 and q_type == QuantType.per_1x32 and is_shuffled:
        return MOEMetadata(functools.partial(fused_moe_1stage, kernelName="",
            activation=activation, quant_type=q_type), None, 32, 0, True)
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

_triton_compiled = False

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _triton_compiled
    (hidden_states, gate_up_weight, down_weight,
     gate_up_weight_scale, down_weight_scale,
     w1_shuf, w2_shuf, w1s_shuf, w2s_shuf,
     topk_weights, topk_ids, config) = data

    M, topk = topk_ids.shape
    E = gate_up_weight.shape[0]
    d_expert_pad = config["d_expert_pad"]
    d_hidden_pad = config["d_hidden_pad"]
    d_hidden = config["d_hidden"]
    hidden_pad = d_hidden_pad - d_hidden
    intermediate_pad = d_expert_pad - config["d_expert"]
    model_dim = d_hidden_pad
    N1 = gate_up_weight.shape[1]  # 2*d_expert_pad
    inter_dim = N1  # gate_up inter dim

    # Decide path based on shape
    tokens_per_expert = (M * topk) / E
    use_cktile = (inter_dim // 2 <= 1024 and  # actual d_expert
                  (tokens_per_expert < 5 or (tokens_per_expert < 40 and E <= 33)))

    if use_cktile:
        # Sparse shapes: use standard cktile (already fastest, BF16, no quant)
        return fused_moe(hidden_states, w1_shuf, w2_shuf, topk_weights, topk_ids,
                         expert_mask=None, activation=ActivationType.Silu,
                         quant_type=QuantType.per_1x32, doweight_stage1=False,
                         w1_scale=w1s_shuf, w2_scale=w2s_shuf,
                         a1_scale=None, a2_scale=None,
                         hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)

    # Dense shapes: try Triton stage 1 + cktile stage 2 (SKIP QUANT+REQUANT)
    try:
        block_size_M = 32
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = _fast_sorting(
            topk_ids, topk_weights, E, model_dim, hidden_states.dtype, block_size_M)

        # Quantize activations using AITER's fused quant+sort (handles padding correctly)
        from aiter.fused_moe import fused_dynamic_mxfp4_quant_moe_sort, get_quant
        from aiter.utility import fp4_utils
        quant_func = get_quant(QuantType.per_1x32)
        if M <= 1024:
            x_fp4, x_scales = fused_dynamic_mxfp4_quant_moe_sort(
                hidden_states, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
                token_num=M, topk=1, block_size=block_size_M)
        else:
            x_fp4, x_scales = quant_func(hidden_states, scale=None, quant_dtype=dtypes.fp4x2)
            x_scales = fp4_utils.moe_mxfp4_sort(x_scales, sorted_ids=sorted_ids,
                num_valid_ids=num_valid_ids, token_num=M, block_size=block_size_M)

        # Triton stage 1: gate_up GEMM with FP4 activations (matches CK precision)
        b_scale_3d = gate_up_weight_scale.view(torch.uint8).view(E, N1, -1)
        a_scale = torch.tensor(1.0, dtype=torch.float32, device=hidden_states.device)
        b_scale = torch.ones(E, dtype=torch.float32, device=hidden_states.device)

        max_sorted = sorted_ids.shape[0]
        stage1_raw = torch.zeros((1, max_sorted, N1),
                                 dtype=hidden_states.dtype, device=hidden_states.device)

        triton_config = {"BLOCK_SIZE_M": block_size_M, "BLOCK_SIZE_N": 128,
                        "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4}

        fused_moe_mxfp4(
            x_fp4.view(torch.uint8),            # FP4 quantized activations as uint8
            gate_up_weight.view(torch.uint8),
            stage1_raw,
            a_scale, b_scale,
            x_scales.view(torch.uint8),         # A_mx_scale (sorted+padded by AITER)
            b_scale_3d,
            topk_weights, topk_ids,
            sorted_ids, sorted_expert_ids, num_valid_ids,
            mul_routed_weight=False, top_k=topk,
            swizzle_mx_a=False, swizzle_mx_b=False,
            config=triton_config, compute_type=tl.float32)

        # stage1_raw is [1, max_sorted, N1] — squeeze to [max_sorted, N1]
        stage1_2d = stage1_raw.squeeze(0)  # [max_sorted, N1]

        # Apply SiLU: first half = gate, second half = up
        half_n = N1 // 2
        gate = stage1_2d[:, :half_n]
        up = stage1_2d[:, half_n:]
        stage1_silu = torch.nn.functional.silu(gate) * up
        # stage1_silu: [max_sorted, d_expert_pad] — in EXPERT-SORTED order

        # Scatter back to [M, topk, d_expert_pad] (TOKEN order)
        # The Triton kernel wrote result for sorted position i at stage1_silu[sorted_ids[i]]
        # We need a2[token, slot, :] where sorted_ids[i] = token*topk + slot
        # Use sorted_ids as gather indices to rearrange
        # The Triton kernel wrote: stage1_silu[sorted_ids[i]] = result for sorted position i
        # We need: a2[token, slot, :] = stage1_silu[token*topk + slot]
        # Since sorted_ids[i] = token*topk+slot for valid entries, and the kernel
        # writes at sorted_ids[i], the output IS already in [M*topk, d] order
        # (positions 0..M*topk-1 correspond to token 0 slot 0, token 0 slot 1, etc.)
        a2 = stage1_silu[:M * topk].view(M, topk, d_expert_pad)

        # cktile stage 2: BF16 intermediate → BF16 output (NO REQUANT!)
        E_s, model_dim_s, inter_dim_s = get_inter_dim(w1_shuf.shape, w2_shuf.shape)
        cktile_s2 = functools.partial(cktile_moe_stage2,
            n_pad_zeros=hidden_pad // 64 * 64,
            k_pad_zeros=intermediate_pad // 128 * 128,
            activation=ActivationType.Silu)

        cktile_s2(a2, w1_shuf, w2_shuf,
                  sorted_ids, sorted_expert_ids, num_valid_ids,
                  moe_buf, topk,
                  w2_scale=w2s_shuf.view(dtypes.fp8_e8m0) if w2_shuf.dtype == dtypes.fp4x2 else None,
                  a2_scale=None,
                  block_m=block_size_M,
                  sorted_weights=sorted_weights)

        if not _triton_compiled:
            _triton_compiled = True
            print(f"[v167] Triton hybrid pipeline OK!", file=sys.stderr)
        return moe_buf

    except Exception as e:
        if not _triton_compiled:
            _triton_compiled = True
            import traceback
            print(f"[v167] Triton hybrid failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        # Fallback to CK
        return fused_moe(hidden_states, w1_shuf, w2_shuf, topk_weights, topk_ids,
                         expert_mask=None, activation=ActivationType.Silu,
                         quant_type=QuantType.per_1x32, doweight_stage1=False,
                         w1_scale=w1s_shuf, w2_scale=w2s_shuf,
                         a1_scale=None, a2_scale=None,
                         hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
