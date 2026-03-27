"""
v165: Minimal custom Triton tl.dot_scaled test.
Writes a tiny GEMM kernel from scratch to verify tl.dot_scaled works
with our AITER fp4x2 weights. If this passes, AITER's Triton kernels
have a bug; if it fails, the weight format is incompatible.
Falls back to CK for actual output.
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

import torch, functools, aiter, triton, triton.language as tl
from task import input_t, output_t
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, get_2stage_cfgs, cktile_moe_stage1, cktile_moe_stage2, fused_moe_1stage, MOEMetadata
import aiter.fused_moe as _fm

# Minimal Triton kernel: single expert GEMM with tl.dot_scaled
@triton.jit
def _test_dot_scaled_kernel(
    A_ptr, B_ptr, C_ptr,
    A_scale_ptr, B_scale_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_asm, stride_ask,
    stride_bsk, stride_bsn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K // 2)  # FP4: 2 per byte
    offs_ks = tl.arange(0, BLOCK_K // 32)  # scales: 1 per 32 elements

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a_scale_ptrs = A_scale_ptr + offs_m[:, None] * stride_asm + offs_ks[None, :] * stride_ask
    # b_scale is [N, K//32], so index as [offs_n, offs_ks]
    b_scale_ptrs = B_scale_ptr + offs_n[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K // BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M)
        b = tl.load(b_ptrs)
        a_scales = tl.load(a_scale_ptrs, mask=offs_m[:, None] < M)
        b_scales = tl.load(b_scale_ptrs)

        acc = tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1", acc=acc)

        a_ptrs += (BLOCK_K // 2) * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk
        a_scale_ptrs += (BLOCK_K // 32) * stride_ask
        b_scale_ptrs += (BLOCK_K // 32) * stride_bsk

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

_tested_triton = False

# Standard v161 patching
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

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _tested_triton
    (hidden_states, gate_up_weight, down_weight,
     gate_up_weight_scale, down_weight_scale,
     w1, w2, w1s, w2s, topk_weights, topk_ids, config) = data

    # One-time test: run minimal tl.dot_scaled on expert 0's weight
    if not _tested_triton:
        _tested_triton = True
        try:
            E = gate_up_weight.shape[0]
            N_w = gate_up_weight.shape[1]  # 2*d_expert_pad
            Kh = gate_up_weight.shape[2]   # d_hidden_pad // 2
            K = Kh * 2                      # logical K

            # Take expert 0's weight and a few rows of hidden_states
            M_test = min(4, hidden_states.shape[0])
            x = hidden_states[:M_test, :]  # [M_test, d_hidden]

            # Quant x to FP4
            from aiter.ops.triton.moe.moe_op_gemm_a4w4 import mxfp4_quant
            x_fp4, x_scales = mxfp4_quant(x)  # [M_test, K//2], [M_test, K//32]

            # Expert 0 weight and scale
            w_e0 = gate_up_weight[0].view(torch.uint8)  # [N_w, K//2]
            # Scale: [E*N, K//32] → expert 0 = first N_w rows
            ws_e0 = gate_up_weight_scale.view(torch.uint8)[:N_w, :]  # [N_w, K//32]

            # Run custom kernel: x_fp4 [M, K//2] @ w_e0^T [K//2, N] = [M, N]
            # Need w in [K//2, N] format. w_e0 is [N, K//2].
            # tl.dot_scaled expects: a=[M, K//2], b=[K//2, N]
            # tl.dot_scaled(a, a_sc, "e2m1", b, b_sc, "e2m1") computes a @ b
            # a: [M, K//2], a_sc: [M, K//32]
            # b: [K//2, N], b_sc: [N, K//32]  ← NOTE: b_scale is [N, K//32] NOT [K//32, N]!
            w_t = w_e0.t().contiguous()  # [K//2, N]
            # ws_e0 is [N, K//32] — DON'T transpose! tl.dot_scaled expects [N, K//32]
            ws_t = ws_e0.contiguous()  # [N, K//32] — keep as-is!

            BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 256
            if M_test <= BLOCK_M and N_w >= BLOCK_N and K >= BLOCK_K:
                out = torch.zeros((M_test, BLOCK_N), dtype=torch.bfloat16, device=x.device)
                grid = (1, 1)
                _test_dot_scaled_kernel[grid](
                    x_fp4, w_t, out, x_scales, ws_t,
                    M_test, BLOCK_N, K,
                    x_fp4.stride(0), x_fp4.stride(1),
                    w_t.stride(0), w_t.stride(1),
                    out.stride(0), out.stride(1),
                    x_scales.stride(0), x_scales.stride(1),
                    ws_t.stride(1), ws_t.stride(0),  # bsk=K//32_stride=1, bsn=N_stride=K//32
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                )

                # Compare against reference (dequant + matmul)
                from aiter.utility.fp4_utils import mxfp4_to_f32, e8m0_to_f32
                x_f32 = mxfp4_to_f32(x_fp4)
                xs_f32 = e8m0_to_f32(x_scales).repeat_interleave(32, dim=-1)[:, :K]
                x_dq = (x_f32 * xs_f32).to(torch.bfloat16)

                w_f32 = mxfp4_to_f32(w_e0)
                ws_f32 = e8m0_to_f32(ws_e0).repeat_interleave(32, dim=-1)[:, :K]
                w_dq = (w_f32 * ws_f32).to(torch.bfloat16)

                ref = x_dq @ w_dq[:BLOCK_N, :].t()
                max_err = (out - ref).abs().max().item()
                print(f"[v165] tl.dot_scaled test: max_err={max_err:.6f} "
                      f"out[0,0]={out[0,0].item():.4f} ref[0,0]={ref[0,0].item():.4f}",
                      file=sys.stderr)
            else:
                print(f"[v165] Shape too small for test: M={M_test} N={N_w} K={K}", file=sys.stderr)
        except Exception as e:
            import traceback
            print(f"[v165] tl.dot_scaled test failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    # Use standard fused_moe for actual output
    return fused_moe(hidden_states, w1, w2, topk_weights, topk_ids,
                     expert_mask=None, activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32, doweight_stage1=False,
                     w1_scale=w1s, w2_scale=w2s, a1_scale=None, a2_scale=None,
                     hidden_pad=config["d_hidden_pad"]-config["d_hidden"],
                     intermediate_pad=config["d_expert_pad"]-config["d_expert"])
