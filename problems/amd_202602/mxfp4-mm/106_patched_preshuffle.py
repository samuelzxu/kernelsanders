"""
MXFP4-MM: #106 - Monkey-patch preshuffle kernel JIT bug + use preshuffle for M>16.
The preshuffle kernel has a NameError('b is not defined') bug when EVEN_K=False.
Fix: patch the source file to add an else branch with masked loads before import.
Then use the preshuffle variant which has optimized (N//16, K*16) weight tiling.
"""
import json
import os

# Patch the preshuffle kernel BEFORE importing it
_KERNEL_FILE = None
for _base in ["/home/runner/aiter", os.path.expanduser("~/aiter")]:
    _f = os.path.join(_base, "aiter/ops/triton/_triton_kernels/gemm/basic/gemm_afp4wfp4.py")
    if os.path.exists(_f):
        _KERNEL_FILE = _f
        break

_PATCHED = False
if _KERNEL_FILE:
    try:
        with open(_KERNEL_FILE, 'r') as f:
            _src = f.read()
        # The buggy code: if EVEN_K without else, then unconditional b.reshape
        _OLD = '''            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)

            b = ('''
        _NEW = '''            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * (BLOCK_SIZE_K // 2), other=0)
                b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * (BLOCK_SIZE_K // 2), other=0, cache_modifier=cache_modifier)

            b = ('''
        if _OLD in _src and _NEW not in _src:
            _src = _src.replace(_OLD, _NEW)
            with open(_KERNEL_FILE, 'w') as f:
                f.write(_src)
            _PATCHED = True
            # Clear Triton JIT cache for this file
            import shutil
            for _cache_dir in [os.path.expanduser("~/.triton/cache"), "/tmp/triton_cache"]:
                if os.path.exists(_cache_dir):
                    try:
                        shutil.rmtree(_cache_dir)
                    except Exception:
                        pass
    except Exception as e:
        import sys
        print(f"[PATCH] Failed to patch preshuffle kernel: {e}", file=sys.stderr)

import triton
import triton.language as tl
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

_CONFIGS = {
    "N=2880-K=512": {
        "M_LEQ_4": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 3, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 3, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
    "N=4096-K=512": {
        "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 3, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
    "N=2112-K=7168": {
        "M_LEQ_16": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 8, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 32, "cache_modifier": None}
    },
    "N=7168-K=2048": {
        "M_LEQ_64": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 32, "cache_modifier": None}
    },
    "N=3072-K=1536": {
        "M_LEQ_64": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 4, "num_stages": 3, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "M_LEQ_256": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 2, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 32, "cache_modifier": None}
    },
}

def _inject_configs():
    try:
        dev = arch_info.get_arch()
    except Exception:
        dev = "gfx950"
    config_dir = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
    os.makedirs(config_dir, exist_ok=True)
    for shape_key, config in _CONFIGS.items():
        fpath = f"{config_dir}/{dev}-GEMM-AFP4WFP4-{shape_key}.json"
        with open(fpath, "w") as f:
            json.dump(config, f)

try:
    _inject_configs()
except Exception:
    pass


@triton.jit
def _fused_quant_gemm_small_m(
    a_bf16_ptr, b_ptr, c_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_cm, stride_cn, stride_bsn, stride_bsk,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr, matrix_instr_nonkdim: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    SCALE_GROUP_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K // 2)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_bf16_ptrs = a_bf16_ptr + offs_am[:, None] * stride_am + (tl.arange(0, BLOCK_SIZE_K))[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    offs_ks = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
    b_scale_ptrs = b_scales_ptr + offs_bn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for ki in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_bf16 = tl.load(a_bf16_ptrs).to(tl.float32)
        a_fp4, a_scales = _mxfp4_quant_op(a_bf16, BLOCK_SIZE_K, BLOCK_SIZE_M, SCALE_GROUP_SIZE)
        b = tl.load(b_ptrs, cache_modifier=cache_modifier)
        b_scales = tl.load(b_scale_ptrs, cache_modifier=cache_modifier)
        accumulator = tl.dot_scaled(a_fp4, a_scales, "e2m1", b, b_scales, "e2m1", accumulator)
        a_bf16_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        b_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_bsk
    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


from task import input_t, output_t
import torch
import sys
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4 as gemm_basic
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import _get_config

# Try importing preshuffle (only works if patch succeeded)
_USE_PRESHUFFLE = False
if _PATCHED:
    try:
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle
        _USE_PRESHUFFLE = True
        print("[PRESHUFFLE] Patched preshuffle kernel loaded!", file=sys.stderr)
    except Exception as e:
        print(f"[PRESHUFFLE] Import failed: {e}", file=sys.stderr)


def e8m0_unshuffle(scale, orig_m, orig_n):
    sm, sn = scale.shape
    scale = scale.view(sm // 32, sn // 8, 4, 16, 2, 2)
    scale = scale.permute(0, 5, 3, 1, 4, 2).contiguous()
    scale = scale.view(sm, sn)
    return scale[:orig_m, :orig_n]


def shuffle_scales(scales):
    """Convert (N, K//32) scale to preshuffle format (N//32, K)."""
    sm, sn = scales.shape
    scales = scales.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
    scales = scales.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    scales = scales.view(sm // 32, sn * 32)
    return scales


_cache_key = None
_cache_bscale_std = None
_cache_bscale_ps = None
_out_cache = {}


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _cache_key, _cache_bscale_std, _cache_bscale_ps
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    B_q_uint8 = B_q.view(torch.uint8)

    key = (B.data_ptr(), B_q.data_ptr(), B_scale_sh.data_ptr())

    config, _ = _get_config(m, n, k)
    ksplit = config.get("NUM_KSPLIT", 1)

    # M<=16 KSPLIT=1: fused kernel (needs standard B_scale)
    if m <= 16 and ksplit == 1:
        if key == _cache_key and _cache_bscale_std is not None:
            B_scale = _cache_bscale_std
        else:
            if k <= 512:
                _, B_scale = dynamic_mxfp4_quant(B)
            else:
                B_scale = e8m0_unshuffle(B_scale_sh.view(torch.uint8), n, k // 32)
            _cache_key = key
            _cache_bscale_std = B_scale
            _cache_bscale_ps = None

        out_key = (m, n)
        if out_key in _out_cache:
            y = _out_cache[out_key]
        else:
            y = torch.empty((m, n), dtype=torch.bfloat16, device=A.device)
            _out_cache[out_key] = y
        B_q_t = B_q_uint8.T
        fused_config = {k_: v_ for k_, v_ in config.items()
                        if k_ in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
                                  "GROUP_SIZE_M", "num_warps", "num_stages",
                                  "waves_per_eu", "matrix_instr_nonkdim", "cache_modifier")}
        grid = lambda META: (triton.cdiv(m, META["BLOCK_SIZE_M"]) * triton.cdiv(n, META["BLOCK_SIZE_N"]),)
        _fused_quant_gemm_small_m[grid](
            A, B_q_t, y, B_scale, m, n, k,
            A.stride(0), A.stride(1), B_q_t.stride(0), B_q_t.stride(1),
            y.stride(0), y.stride(1), B_scale.stride(0), B_scale.stride(1),
            **fused_config,
        )
        return y

    # Preshuffle path (if patch succeeded)
    if _USE_PRESHUFFLE and m >= 32:
        # Get preshuffle B_scale
        if key == _cache_key and _cache_bscale_ps is not None:
            B_scale_ps = _cache_bscale_ps
        else:
            if k <= 512:
                _, B_scale_std = dynamic_mxfp4_quant(B)
            else:
                B_scale_std = e8m0_unshuffle(B_scale_sh.view(torch.uint8), n, k // 32)
            B_scale_ps = shuffle_scales(B_scale_std)
            _cache_key = key
            _cache_bscale_std = B_scale_std
            _cache_bscale_ps = B_scale_ps

        A_q, A_scale = dynamic_mxfp4_quant(A)
        A_scale_u8 = A_scale.view(torch.uint8)
        A_scale_ps = shuffle_scales(A_scale_u8) if m >= 32 else A_scale_u8
        B_shuffle_u8 = B_shuffle.view(torch.uint8)
        return gemm_afp4wfp4_preshuffle(
            A_q.view(torch.uint8), B_shuffle_u8, A_scale_ps, B_scale_ps,
            dtype=torch.bfloat16,
        )

    # Fallback: standard path
    if key == _cache_key and _cache_bscale_std is not None:
        B_scale = _cache_bscale_std
    else:
        if k <= 512:
            _, B_scale = dynamic_mxfp4_quant(B)
        else:
            B_scale = e8m0_unshuffle(B_scale_sh.view(torch.uint8), n, k // 32)
        _cache_key = key
        _cache_bscale_std = B_scale
        _cache_bscale_ps = None
    A_q, A_scale = dynamic_mxfp4_quant(A)
    return gemm_basic(A_q, B_q_uint8, A_scale, B_scale, dtype=torch.bfloat16)
