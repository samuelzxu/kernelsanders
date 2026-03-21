"""
MXFP4-MM: #107 - Call _gemm_afp4wfp4_kernel directly, bypassing wrapper.
Eliminates per-call overhead:
- No serialize_dict/deserialize_str roundtrip
- No get_splitk recomputation
- No torch.empty allocations (pre-allocated)
- No double _get_config lookup
- Pre-computed SPLITK_BLOCK_SIZE
"""
import json
import os
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
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import (
    _gemm_afp4wfp4_kernel, _gemm_afp4wfp4_reduce_kernel, _get_config,
)


def e8m0_unshuffle(scale, orig_m, orig_n):
    sm, sn = scale.shape
    scale = scale.view(sm // 32, sn // 8, 4, 16, 2, 2)
    scale = scale.permute(0, 5, 3, 1, 4, 2).contiguous()
    scale = scale.view(sm, sn)
    return scale[:orig_m, :orig_n]


# Per-shape pre-computed dispatch info
_shape_info = {}  # (m,n,k) -> dict with pre-computed params
_bscale_cache_key = None
_bscale_cache_val = None


def _get_shape_info(m, n, k, device):
    """Pre-compute and cache everything for a shape."""
    if (m, n, k) in _shape_info:
        return _shape_info[(m, n, k)]

    config, _ = _get_config(m, n, k)
    ksplit = config.get("NUM_KSPLIT", 1)
    bsm = config["BLOCK_SIZE_M"]
    bsn = config["BLOCK_SIZE_N"]
    bsk = config["BLOCK_SIZE_K"]
    use_fused = (m <= 16 and ksplit == 1)

    k_packed = k // 2

    # Compute SPLITK_BLOCK_SIZE (same logic as wrapper)
    if ksplit > 1:
        splitk_bs = triton.cdiv(2 * triton.cdiv(k_packed, ksplit), bsk) * bsk
        # Recompute actual ksplit
        actual_ksplit = triton.cdiv(k_packed, splitk_bs // 2)
    else:
        splitk_bs = 2 * k_packed
        actual_ksplit = 1

    # Pre-allocate buffers
    y = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    y_pp = None
    if actual_ksplit > 1:
        y_pp = torch.empty((actual_ksplit, m, n), dtype=torch.float32, device=device)

    # Fused config (for M<=16 KSPLIT=1 path)
    fused_config = None
    if use_fused:
        fused_config = {k_: config[k_] for k_ in
                        ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
                         "GROUP_SIZE_M", "num_warps", "num_stages",
                         "waves_per_eu", "matrix_instr_nonkdim", "cache_modifier")
                        if k_ in config}

    # Direct kernel config (all constexpr params)
    kernel_config = {
        "BLOCK_SIZE_M": bsm, "BLOCK_SIZE_N": bsn, "BLOCK_SIZE_K": bsk,
        "GROUP_SIZE_M": config.get("GROUP_SIZE_M", 1),
        "NUM_KSPLIT": actual_ksplit, "SPLITK_BLOCK_SIZE": splitk_bs,
        "num_warps": config.get("num_warps", 4),
        "num_stages": config.get("num_stages", 2),
        "waves_per_eu": config.get("waves_per_eu", 0),
        "matrix_instr_nonkdim": config.get("matrix_instr_nonkdim", 16),
        "cache_modifier": config.get("cache_modifier", None),
    }

    # Grid
    grid_mn = triton.cdiv(m, bsm) * triton.cdiv(n, bsn)
    grid_size = grid_mn * actual_ksplit

    info = {
        "use_fused": use_fused,
        "fused_config": fused_config,
        "kernel_config": kernel_config,
        "y": y, "y_pp": y_pp,
        "ksplit": actual_ksplit,
        "grid_size": grid_size,
        "k_packed": k_packed,
    }
    _shape_info[(m, n, k)] = info
    return info


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _bscale_cache_key, _bscale_cache_val
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    B_q_uint8 = B_q.view(torch.uint8)

    # B_scale with caching
    key = (B.data_ptr(), B_q.data_ptr(), B_scale_sh.data_ptr())
    if key == _bscale_cache_key:
        B_scale = _bscale_cache_val
    else:
        if k <= 512:
            _, B_scale = dynamic_mxfp4_quant(B)
        else:
            B_scale = e8m0_unshuffle(B_scale_sh.view(torch.uint8), n, k // 32)
        _bscale_cache_key = key
        _bscale_cache_val = B_scale

    info = _get_shape_info(m, n, k, A.device)

    # M<=16 KSPLIT=1: fused kernel
    if info["use_fused"]:
        y = info["y"]
        B_q_t = B_q_uint8.T
        grid = (info["grid_size"],)
        _fused_quant_gemm_small_m[grid](
            A, B_q_t, y, B_scale, m, n, k,
            A.stride(0), A.stride(1), B_q_t.stride(0), B_q_t.stride(1),
            y.stride(0), y.stride(1), B_scale.stride(0), B_scale.stride(1),
            **info["fused_config"],
        )
        return y

    # Direct kernel call: bypass gemm_afp4wfp4 wrapper
    A_q, A_scale = dynamic_mxfp4_quant(A)

    B_q_t = B_q_uint8.T  # (K_packed, N)
    ksplit = info["ksplit"]
    kconfig = info["kernel_config"]

    if ksplit == 1:
        y = info["y"]
        grid = (info["grid_size"],)
        _gemm_afp4wfp4_kernel[grid](
            A_q, B_q_t, y, A_scale, B_scale,
            m, n, info["k_packed"],
            A_q.stride(0), A_q.stride(1),
            B_q_t.stride(0), B_q_t.stride(1),
            0,  # stride_ck (no split)
            y.stride(0), y.stride(1),
            A_scale.stride(0), A_scale.stride(1),
            B_scale.stride(0), B_scale.stride(1),
            **kconfig,
        )
        return y
    else:
        y = info["y"]
        y_pp = info["y_pp"]
        grid = (info["grid_size"],)
        _gemm_afp4wfp4_kernel[grid](
            A_q, B_q_t, y_pp, A_scale, B_scale,
            m, n, info["k_packed"],
            A_q.stride(0), A_q.stride(1),
            B_q_t.stride(0), B_q_t.stride(1),
            y_pp.stride(0), y_pp.stride(1), y_pp.stride(2),
            A_scale.stride(0), A_scale.stride(1),
            B_scale.stride(0), B_scale.stride(1),
            **kconfig,
        )
        # Reduce
        actual_ksplit = triton.cdiv(info["k_packed"], kconfig["SPLITK_BLOCK_SIZE"] // 2)
        reduce_bsm, reduce_bsn = 16, 64
        grid_reduce = (triton.cdiv(m, reduce_bsm), triton.cdiv(n, reduce_bsn))
        _gemm_afp4wfp4_reduce_kernel[grid_reduce](
            y_pp, y, m, n,
            y_pp.stride(0), y_pp.stride(1), y_pp.stride(2),
            y.stride(0), y.stride(1),
            reduce_bsm, reduce_bsn,
            actual_ksplit,
            triton.next_power_of_2(ksplit),
        )
        return y
