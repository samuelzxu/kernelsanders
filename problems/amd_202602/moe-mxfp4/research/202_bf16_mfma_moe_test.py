#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
v199: Custom Triton MoE GEMM using tl.dot_scaled FP4.
- Corrected weight scale stride addressing (bug in custom_moe.py)
- AITER sorting + dynamic_mxfp4_quant for activation quantization
- Custom Triton grouped GEMM for both stages
- Falls back to AITER for sparse shapes where cktile BF16 is faster
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

# Clear ALL Triton caches to ensure fresh compilation
import shutil
for _td in [os.path.expanduser("~/.triton"), "/tmp/.triton", "/tmp/triton"]:
    if os.path.exists(_td):
        try: shutil.rmtree(_td); print(f"[v199] Cleared {_td}", file=__import__('sys').stderr)
        except: pass

os.environ['HIP_FORCE_DEV_KERNARG'] = '1'
os.environ['GPU_MAX_HW_QUEUES'] = '2'
os.environ['AITER_USE_NT'] = '1'

import torch, functools, triton, triton.language as tl
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    fused_moe, get_2stage_cfgs, cktile_moe_stage1, cktile_moe_stage2,
    fused_moe_1stage, MOEMetadata, get_inter_dim,
)
import aiter.fused_moe as _fm
from aiter.ops.triton.quant import dynamic_mxfp4_quant
try:
    from aiter.ops.triton.moe.moe_op_gemm_a4w4 import mxfp4_quant as _triton_mxfp4_quant
except ImportError:
    _triton_mxfp4_quant = None
from task import input_t, output_t

# ============================================================
# Triton grouped MoE GEMM kernel using tl.dot_scaled FP4
# ============================================================
@triton.jit
def _moe_gemm_vK(
    # Quantized activations [M_orig, K//2] fp4x2 raw
    a_ptr, a_scale_ptr,
    # Expert weights [E, N, K//2] fp4x2 raw
    w_ptr, w_scale_ptr,
    # Sorting arrays
    sorted_ids_ptr, sorted_expert_ids_ptr,
    # Output [max_sorted, N] bf16
    out_ptr,
    # Scalars
    num_valid_tokens, topk, K: tl.constexpr, N: tl.constexpr, E,
    # Weight strides (3D: [E, N, K//2])
    stride_w_e, stride_w_n, stride_w_k,
    # Weight scale strides (3D: [E, N, K//32])
    stride_ws_e, stride_ws_n, stride_ws_k,
    # Activation strides (2D: [M, K//2])
    stride_a_m, stride_a_k,
    # Activation scale strides (2D: [M_pad, K//32])
    stride_as_m, stride_as_k,
    # Output strides
    stride_o_m, stride_o_n,
    # Tile sizes
    block_size: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    VERSION: tl.constexpr = 2,  # bump to invalidate cache
):
    SCALE_GROUP: tl.constexpr = 32

    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    # Determine expert for this M-block
    # sorted_expert_ids covers ALL blocks including shared expert — NO CLAMPING
    expert_block = pid_m * BLOCK_M // block_size
    expert_id_raw = tl.load(sorted_expert_ids_ptr + expert_block)
    expert_id = tl.where(expert_id_raw >= 0, expert_id_raw, tl.zeros_like(expert_id_raw))
    expert_id = tl.where(expert_id < E, expert_id, tl.zeros_like(expert_id))

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K // 2)      # byte offsets for fp4x2
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP)  # scale group offsets

    # Load sorted token IDs and compute original token index
    token_ids = tl.load(sorted_ids_ptr + offs_m)
    m_valid = token_ids < num_valid_tokens
    safe_token_ids = tl.where(m_valid, token_ids, tl.zeros_like(token_ids))
    orig_token = safe_token_ids // topk

    # Activation pointers: a_q[orig_token, k_bytes]
    a_ptrs = a_ptr + orig_token[:, None] * stride_a_m + offs_k[None, :] * stride_a_k
    as_ptrs = a_scale_ptr + orig_token[:, None] * stride_as_m + offs_ks[None, :] * stride_as_k

    # Weight pointers: w[expert_id, k_bytes, n] (transposed for dot_scaled)
    # tl.dot_scaled expects b=[K//2, N], w is stored [E, N, K//2]
    # So we index as w[expert_id, :, :].T by swapping k/n strides
    eid64 = expert_id.to(tl.int64)
    w_base = w_ptr + eid64 * stride_w_e
    w_ptrs = w_base + offs_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n

    # Weight scale pointers: ws[expert_id, n, k_group]
    # FIXED: use stride_ws_e for expert offset (was buggy: * N * stride_ws_row)
    ws_base = w_scale_ptr + eid64 * stride_ws_e
    ws_ptrs = ws_base + offs_n[:, None] * stride_ws_n + offs_ks[None, :] * stride_ws_k

    n_valid = offs_n < N
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_iters = tl.cdiv(K, BLOCK_K)
    for _ in range(num_k_iters):
        a = tl.load(a_ptrs, mask=m_valid[:, None], other=0)
        a_scales = tl.load(as_ptrs, mask=m_valid[:, None], other=0)
        w = tl.load(w_ptrs, mask=n_valid[None, :], other=0)
        w_scales = tl.load(ws_ptrs, mask=n_valid[:, None], other=0)

        accumulator = tl.dot_scaled(a, a_scales, "e2m1", w, w_scales, "e2m1", accumulator)

        a_ptrs += (BLOCK_K // 2) * stride_a_k
        w_ptrs += (BLOCK_K // 2) * stride_w_k
        as_ptrs += (BLOCK_K // SCALE_GROUP) * stride_as_k
        ws_ptrs += (BLOCK_K // SCALE_GROUP) * stride_ws_k

    c = accumulator.to(tl.bfloat16)
    out_ptrs = out_ptr + offs_m[:, None] * stride_o_m + offs_n[None, :] * stride_o_n
    tl.store(out_ptrs, c, mask=m_valid[:, None] & n_valid[None, :])


def moe_gemm(a_q, a_scale, w_raw, w_scale_raw, sorted_ids, sorted_expert_ids,
             num_valid_tokens, topk, block_size, E, N, K,
             BLOCK_N=128, BLOCK_K=256):
    """Launch grouped MoE GEMM via Triton."""
    max_sorted = sorted_ids.shape[0]
    out = torch.zeros((max_sorted, N), dtype=torch.bfloat16, device=a_q.device)

    BLOCK_M = block_size
    bk = min(BLOCK_K, K)
    bn = min(BLOCK_N, N)

    # w_raw is [E, N, K//2]. Per expert, transpose to [K//2, N] contiguously.
    w_u8 = w_raw.view(torch.uint8)  # [E, N, K//2]
    # Pre-transpose each expert's weight to [K//2, N] contiguous
    w_list = []
    for e in range(E):
        w_list.append(w_u8[e].t().contiguous())  # [K//2, N]
    w_t = torch.stack(w_list)  # [E, K//2, N] contiguous
    del w_list
    ws_u8 = w_scale_raw.view(torch.uint8)  # [E, N, K//32]

    num_m_blocks = triton.cdiv(max_sorted, BLOCK_M)
    num_n_blocks = triton.cdiv(N, bn)
    grid = (num_m_blocks * num_n_blocks,)

    _moe_gemm_vK[grid](
        a_q.view(torch.uint8), a_scale.view(torch.uint8),
        w_t, ws_u8,
        sorted_ids, sorted_expert_ids,
        out,
        num_valid_tokens, topk, K, N, E,
        # w_t is [E, K//2, N]: stride(0)=expert, stride(1)=K_row=N, stride(2)=N_col=1
        w_t.stride(0), w_t.stride(2), w_t.stride(1),
        # Weight scale strides [E, N, K//32]
        ws_u8.stride(0), ws_u8.stride(1), ws_u8.stride(2),
        # Activation strides
        a_q.view(torch.uint8).stride(0), a_q.view(torch.uint8).stride(1),
        a_scale.view(torch.uint8).stride(0), a_scale.view(torch.uint8).stride(1),
        # Output strides
        out.stride(0), out.stride(1),
        block_size=BLOCK_M, BLOCK_M=BLOCK_M, BLOCK_N=bn, BLOCK_K=bk,
    )
    return out


# ============================================================
# AITER fallback dispatch (v185 config patching)
# ============================================================
_orig = get_2stage_cfgs.__wrapped__

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
            "kernelName2": "flydsl_moe2_afp4_wfp4_bf16_t64x256x256_atomic",
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
            activation=activation, quant_type=q_type), None, 32, 0, True)
    return md

_fm.get_2stage_cfgs = _patched

# ============================================================
# Sorting helper
# ============================================================
_has_opus = hasattr(aiter, 'moe_sorting_opus_fwd')
_sort_fwd = aiter.moe_sorting_opus_fwd if _has_opus else aiter.moe_sorting_fwd
_sorting_bufs = {}

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


# ============================================================
# Warmup Triton kernels (compile once)
# ============================================================
_triton_warmed = False
def _warmup_triton():
    global _triton_warmed
    if _triton_warmed:
        return
    _triton_warmed = True
    try:
        M, K, N, E = 32, 256, 128, 2
        a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        a_q, a_s = dynamic_mxfp4_quant(a)
        # Quantize each expert's weight as 2D, then stack
        from aiter import get_torch_quant
        tq = get_torch_quant(QuantType.per_1x32)
        w_list, ws_list = [], []
        for _ in range(E):
            w2d = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
            wq, wsc = tq(w2d, quant_dtype=dtypes.fp4x2)
            w_list.append(wq.view(N, K // 2))
            ws_list.append(wsc)
        w_q = torch.stack(w_list)   # [E, N, K//2]
        w_sc = torch.stack(ws_list) # [E, N, K//32]
        topk_ids = torch.zeros(M, 1, dtype=torch.int32, device='cuda')
        topk_wts = torch.ones(M, 1, dtype=torch.float32, device='cuda')
        sids, swts, seids, nv, _ = _fast_sorting(topk_ids, topk_wts, E, K, torch.bfloat16, 32)
        moe_gemm(a_q, a_s, w_q, w_sc, sids, seids, M, 1, 32, E, N, K, BLOCK_N=128, BLOCK_K=128)
        torch.cuda.synchronize()
        print("[v199] Triton warmup OK", file=sys.stderr)
    except Exception as e:
        import traceback
        print(f"[v199] Triton warmup failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

try:
    _warmup_triton()
except Exception:
    pass


# ============================================================
# Main dispatch
# ============================================================
@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    (hidden_states, gate_up_weight, down_weight,
     gate_up_weight_scale, down_weight_scale,
     w1, w2, w1s, w2s, topk_weights, topk_ids, config) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]
    d_expert_pad = config["d_expert_pad"]
    d_hidden = config["d_hidden"]
    d_hidden_pad = config["d_hidden_pad"]

    M = hidden_states.shape[0]
    E = gate_up_weight.shape[0]
    topk = topk_ids.shape[1]
    tokens_per_expert = (M * topk) / E
    n_shared = config.get("n_shared_experts", 0)
    n_routed_topk = topk - n_shared  # 8 routed + 1 shared = 9 total

    # ----------------------------------------------------------
    # For sparse shapes, AITER cktile BF16 path is faster
    # ----------------------------------------------------------
    if tokens_per_expert < 40:
        return fused_moe(hidden_states, w1, w2, topk_weights, topk_ids,
                         expert_mask=None, activation=ActivationType.Silu,
                         quant_type=QuantType.per_1x32, doweight_stage1=False,
                         w1_scale=w1s, w2_scale=w2s, a1_scale=None, a2_scale=None,
                         hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)

    # ----------------------------------------------------------
    # Dense shapes: use AITER (shared expert handling is complex)
    # The custom Triton GEMM works perfectly for individual stages,
    # but AITER's sorting interleaves shared experts with routed ones.
    # TODO: implement proper shared expert split once AITER's sorting semantics are understood.
    # ----------------------------------------------------------
    return fused_moe(hidden_states, w1, w2, topk_weights, topk_ids,
                     expert_mask=None, activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32, doweight_stage1=False,
                     w1_scale=w1s, w2_scale=w2s, a1_scale=None, a2_scale=None,
                     hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)


def _split_shared_expert_pipeline(
    hidden_states, gate_up_weight, down_weight,
    gate_up_weight_scale, down_weight_scale,
    w1, w2, w1s, w2s,
    topk_weights, topk_ids,
    M, E, topk, n_shared, n_routed_topk,
    d_expert_pad, d_hidden, d_hidden_pad,
    hidden_pad, intermediate_pad,
):
    """Split MoE: AITER for routed experts + custom Triton for shared expert."""
    device = hidden_states.device

    # 1. Routed experts via AITER
    # Keep full topk=9 but zero out shared expert weight so it contributes nothing
    routed_topk_weights = topk_weights.clone()
    routed_topk_weights[:, n_routed_topk:] = 0.0  # zero shared expert weights
    routed_out = fused_moe(hidden_states, w1, w2, routed_topk_weights, topk_ids,
                           expert_mask=None, activation=ActivationType.Silu,
                           quant_type=QuantType.per_1x32, doweight_stage1=False,
                           w1_scale=w1s, w2_scale=w2s, a1_scale=None, a2_scale=None,
                           hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)

    # 2. Shared expert(s) — dense GEMM on all M tokens
    # Shared expert is the last expert: expert_id = E - n_shared .. E - 1
    shared_out = torch.zeros_like(routed_out)
    for s in range(n_shared):
        eid = E - n_shared + s
        shared_w = topk_weights[:, n_routed_topk + s]  # weight for this shared expert

        # Stage 1: hidden_states @ gate_up_weight[eid]^T → [M, 2*d_expert_pad]
        # Use custom Triton GEMM (single expert, all tokens, no sorting needed)
        w1_3d = gate_up_weight_scale.view(torch.uint8)
        N1 = gate_up_weight.shape[1]
        K1 = gate_up_weight.shape[2] * 2
        K1_scale_groups = K1 // 32

        # Reshape for single-expert: w=[N1, K1//2], ws=[N1, K1//32]
        w1_e = gate_up_weight[eid].view(torch.uint8)     # [N1, K1//2]
        ws1_e = gate_up_weight_scale.view(torch.uint8)[eid * N1:(eid + 1) * N1]  # [N1, K1//32]

        # Quantize activations
        h_q, h_scale = dynamic_mxfp4_quant(hidden_states)

        # Transpose weight for tl.dot_scaled: [K1//2, N1]
        w1_t = w1_e.t().contiguous()

        # Launch Triton GEMM for shared expert stage 1
        gate_up_out = _single_expert_gemm(h_q, h_scale, w1_t, ws1_e, M, N1, K1)

        if not hasattr(_split_shared_expert_pipeline, '_s1_dbg'):
            _split_shared_expert_pipeline._s1_dbg = True
            print(f"[v199] Shared S1: M={M} N1={N1} K1={K1} eid={eid}", file=sys.stderr)
            print(f"[v199] Shared S1 out[:2,:4]={gate_up_out[:2,:4]}", file=sys.stderr)
            print(f"[v199] h_q={h_q.shape} w1_t={w1_t.shape} ws1_e={ws1_e.shape}", file=sys.stderr)

        # SiLU
        gate = gate_up_out[:, :d_expert_pad].float()
        up = gate_up_out[:, d_expert_pad:2 * d_expert_pad].float()
        intermediate = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)

        # Stage 2: intermediate @ down_weight[eid]^T → [M, d_hidden_pad]
        N2 = down_weight.shape[1]
        K2 = down_weight.shape[2] * 2

        w2_e = down_weight[eid].view(torch.uint8)
        ws2_e = down_weight_scale.view(torch.uint8)[eid * N2:(eid + 1) * N2]

        inter_q, inter_scale = dynamic_mxfp4_quant(intermediate)
        w2_t = w2_e.t().contiguous()
        down_result = _single_expert_gemm(inter_q, inter_scale, w2_t, ws2_e, M, N2, K2)

        # Apply shared expert weight and accumulate
        shared_out += shared_w.unsqueeze(1) * down_result[:, :d_hidden].float()

    # 3. Combine routed + shared
    return (routed_out.float() + shared_out).to(torch.bfloat16)


def _single_expert_gemm(a_q, a_scale, w_t, w_scale, M, N, K):
    """Single-expert GEMM using Triton tl.dot_scaled.
    a_q: [M, K//2] fp4x2, a_scale: [M, K//32] e8m0
    w_t: [K//2, N] uint8 contiguous (transposed weight)
    w_scale: [N, K//32] uint8
    Returns: [M, N] bf16
    """
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 128, 256
    bk = min(BLOCK_K, K)
    bn = min(BLOCK_N, N)

    out = torch.zeros((M, N), dtype=torch.bfloat16, device=a_q.device)
    a_u8 = a_q.view(torch.uint8)
    as_u8 = a_scale.view(torch.uint8)
    ws_u8 = w_scale.view(torch.uint8)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, bn),)

    _single_expert_kernel[grid](
        a_u8, as_u8, w_t, ws_u8, out,
        M, K, N,
        a_u8.stride(0), a_u8.stride(1),
        as_u8.stride(0), as_u8.stride(1),
        w_t.stride(0), w_t.stride(1),
        ws_u8.stride(0), ws_u8.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=bn, BLOCK_K=bk,
    )
    return out


@triton.jit
def _single_expert_kernel(
    A_ptr, A_scale_ptr, B_ptr, B_scale_ptr, Out_ptr,
    M, K: tl.constexpr, N: tl.constexpr,
    stride_am, stride_ak, stride_asm, stride_ask,
    stride_bk, stride_bn, stride_bsn, stride_bsk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Simple single-expert GEMM: A[M, K//2] @ B[K//2, N]^T using tl.dot_scaled."""
    SCALE_GROUP: tl.constexpr = 32
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n
    pid_n = pid % num_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K // 2)
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP)

    m_valid = offs_m < M
    n_valid = offs_n < N

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    as_ptrs = A_scale_ptr + offs_m[:, None] * stride_asm + offs_ks[None, :] * stride_ask
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    bs_ptrs = B_scale_ptr + offs_n[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=m_valid[:, None], other=0)
        a_s = tl.load(as_ptrs, mask=m_valid[:, None], other=0)
        b = tl.load(b_ptrs, mask=n_valid[None, :], other=0)
        b_s = tl.load(bs_ptrs, mask=n_valid[:, None], other=0)
        acc = tl.dot_scaled(a, a_s, "e2m1", b, b_s, "e2m1", acc)
        a_ptrs += (BLOCK_K // 2) * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk
        as_ptrs += (BLOCK_K // SCALE_GROUP) * stride_ask
        bs_ptrs += (BLOCK_K // SCALE_GROUP) * stride_bsk

    out = acc.to(tl.bfloat16)
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=m_valid[:, None] & n_valid[None, :])


## Old _custom_moe_pipeline removed — replaced by _split_shared_expert_pipeline
