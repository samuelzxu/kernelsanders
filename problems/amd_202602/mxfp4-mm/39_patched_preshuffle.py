"""
MXFP4-MM: Patched preshuffle kernel + config injection.
Fixes the EVEN_K bug in _gemm_afp4wfp4_preshuffle_kernel.
Uses pre-shuffled B weights + shuffled scales directly - no double quant or unshuffle.
"""
import json
import os
import triton
import triton.language as tl
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import (
    _get_config,
    _gemm_afp4wfp4_reduce_kernel,
)

# Inject K=512 Triton configs
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
}

def _inject_configs():
    try:
        dev = arch_info.get_arch()
    except Exception:
        dev = "gfx950"
    config_dir = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
    os.makedirs(config_dir, exist_ok=True)
    for shape_key, config in _CONFIGS.items():
        fpath = f"{config_dir}/{dev}-GEMM-AFP4WFP4_PRESHUFFLED-{shape_key}.json"
        if not os.path.exists(fpath):
            with open(fpath, "w") as f:
                json.dump(config, f)

try:
    _inject_configs()
except Exception:
    pass


# Patched preshuffle kernel: fixes NameError for EVEN_K=False
@triton.jit
def _patched_preshuffle_kernel(
    a_ptr, b_ptr, c_ptr, a_scales_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bn, stride_bk,
    stride_ck, stride_cm, stride_cn,
    stride_asm, stride_ask, stride_bsn, stride_bsk,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr, SPLITK_BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr, matrix_instr_nonkdim: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_asm > 0)
    tl.assume(stride_ask > 0)
    tl.assume(stride_bsk > 0)
    tl.assume(stride_bsn > 0)

    GRID_MN = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    pid_unified = tl.program_id(axis=0)
    pid_unified = remap_xcd(pid_unified, GRID_MN * NUM_KSPLIT, NUM_XCDS=8)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    SCALE_GROUP_SIZE: tl.constexpr = 32

    if (pid_k * SPLITK_BLOCK_SIZE // 2) < K:
        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE // 2, BLOCK_SIZE_K // 2)

        offs_k = tl.arange(0, BLOCK_SIZE_K // 2)
        offs_k_shuffle_arr = tl.arange(0, (BLOCK_SIZE_K // 2) * 16)
        offs_k_split = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_k
        offs_k_shuffle = pid_k * (SPLITK_BLOCK_SIZE // 2) * 16 + offs_k_shuffle_arr

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * (BLOCK_SIZE_N // 16) + tl.arange(0, BLOCK_SIZE_N // 16)) % N
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + offs_k_shuffle[None, :] * stride_bk)

        offs_asn = (pid_n * (BLOCK_SIZE_N // 32) + tl.arange(0, (BLOCK_SIZE_N // 32))) % N
        offs_ks = (pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE) * 32) + tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE * 32)
        b_scale_ptrs = b_scales_ptr + offs_asn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

        if BLOCK_SIZE_M < 32:
            offs_ks_non_shufl = (pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE)) + tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
            a_scale_ptrs = a_scales_ptr + offs_am[:, None] * stride_asm + offs_ks_non_shufl[None, :] * stride_ask
        else:
            offs_asm = (pid_m * (BLOCK_SIZE_M // 32) + tl.arange(0, (BLOCK_SIZE_M // 32))) % M
            a_scale_ptrs = a_scales_ptr + offs_asm[:, None] * stride_asm + offs_ks[None, :] * stride_ask

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
            if BLOCK_SIZE_M < 32:
                a_scales = tl.load(a_scale_ptrs)
            else:
                a_scales = (
                    tl.load(a_scale_ptrs)
                    .reshape(BLOCK_SIZE_M // 32, BLOCK_SIZE_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2, 1)
                    .permute(0, 5, 3, 1, 4, 2, 6)
                    .reshape(BLOCK_SIZE_M, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                )
            b_scales = (
                tl.load(b_scale_ptrs, cache_modifier=cache_modifier)
                .reshape(BLOCK_SIZE_N // 32, BLOCK_SIZE_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2, 1)
                .permute(0, 5, 3, 1, 4, 2, 6)
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
            )

            # FIX: Always load a and b (the bug was missing else clause)
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs, cache_modifier=cache_modifier)

            b = (
                b.reshape(1, BLOCK_SIZE_N // 16, BLOCK_SIZE_K // 64, 2, 16, 16)
                .permute(0, 1, 4, 2, 3, 5)
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K // 2)
                .trans(1, 0)
            )

            accumulator = tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1", accumulator)

            a_ptrs += (BLOCK_SIZE_K // 2) * stride_ak
            b_ptrs += (BLOCK_SIZE_K // 2) * 16 * stride_bk
            if BLOCK_SIZE_M < 32:
                a_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_ask
            else:
                a_scale_ptrs += BLOCK_SIZE_K * stride_ask
            b_scale_ptrs += BLOCK_SIZE_K * stride_bsk

        c = accumulator.to(c_ptr.type.element_ty)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :] + pid_k * stride_ck
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask, cache_modifier=".wt")


from task import input_t, output_t
import torch
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import get_splitk

global _USE_GEMM_SPLITK_BF16
_USE_GEMM_SPLITK_BF16 = False


def patched_preshuffle(x, w, x_scales, w_scales, dtype=torch.bfloat16, y=None):
    """Wrapper for patched preshuffle kernel."""
    M, K = x.shape
    N, K_raw = w.shape
    N = N * 16
    K = K_raw // 16

    config, _ = _get_config(M, N, K, True)

    if config["NUM_KSPLIT"] > 1:
        SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = get_splitk(K, config["BLOCK_SIZE_K"], config["NUM_KSPLIT"])
        config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
        config["BLOCK_SIZE_K"] = BLOCK_SIZE_K
        config["NUM_KSPLIT"] = NUM_KSPLIT
        y_pp = torch.empty((config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=x.device)
    else:
        config["SPLITK_BLOCK_SIZE"] = 2 * K
        y_pp = None

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if config["BLOCK_SIZE_K"] >= 2 * K:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(2 * K)
        config["SPLITK_BLOCK_SIZE"] = 2 * K

    config["BLOCK_SIZE_N"] = max(config["BLOCK_SIZE_N"], 32)
    if M < 32:
        assert config["BLOCK_SIZE_M"] <= 16
    else:
        assert config["BLOCK_SIZE_M"] >= 32

    grid = lambda META: (META["NUM_KSPLIT"] * triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    _patched_preshuffle_kernel[grid](
        x, w,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        x_scales, w_scales,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
        y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
        y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
        x_scales.stride(0), x_scales.stride(1),
        w_scales.stride(0), w_scales.stride(1),
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 16
        REDUCE_BLOCK_SIZE_N = 64
        ACTUAL_KSPLIT = triton.cdiv(K, (config["SPLITK_BLOCK_SIZE"] // 2))
        grid_reduce = (triton.cdiv(M, REDUCE_BLOCK_SIZE_M), triton.cdiv(N, REDUCE_BLOCK_SIZE_N))
        _gemm_afp4wfp4_reduce_kernel[grid_reduce](
            y_pp, y, M, N,
            y_pp.stride(0), y_pp.stride(1), y_pp.stride(2),
            y.stride(0), y.stride(1),
            REDUCE_BLOCK_SIZE_M, REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT, triton.next_power_of_2(config["NUM_KSPLIT"]),
        )

    return y


_cache_key = None
_cache_bscale = None


def custom_kernel(data: input_t) -> output_t:
    global _cache_key, _cache_bscale
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    A_q, A_scale = dynamic_mxfp4_quant(A)

    # For M >= 32: shuffle A_scale for preshuffle kernel
    # For M < 32: keep A_scale unshuffled
    if m >= 32:
        A_scale_proc = e8m0_shuffle(A_scale)
    else:
        A_scale_proc = A_scale

    # Use preshuffle kernel: takes B_shuffle + B_scale_sh directly
    return patched_preshuffle(
        A_q.view(torch.uint8),
        B_shuffle.view(torch.uint8),
        A_scale_proc.view(torch.uint8),
        B_scale_sh.view(torch.uint8),
        dtype=torch.bfloat16,
    )
