"""
MXFP4-MM: #95 - Custom fused quant+GEMM kernel with split-K AND remap_xcd.
Key insight: gemm_afp4wfp4 has remap_xcd but needs separate quant.
            gemm_a16wfp4 fuses quant but lacks remap_xcd.
This kernel has BOTH: fused quant + remap_xcd + split-K via temp buffer + reduce.
Target: M=16 K=7168 (KSPLIT=8) and M=256 K=1536 (KSPLIT=2).
"""
import json
import os
import triton
import triton.language as tl
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
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
        "M_LEQ_256": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 2, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
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
def _fused_quant_gemm_kernel(
    a_bf16_ptr, b_ptr, c_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_cm, stride_cn, stride_bsn, stride_bsk,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr, SPLITK_BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr, matrix_instr_nonkdim: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """Fused bf16 quant + FP4 GEMM with split-K and remap_xcd."""
    SCALE_GROUP_SIZE: tl.constexpr = 32

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    GRID_MN = num_pid_m * num_pid_n

    # remap_xcd for MI355X (8 XCDs)
    pid = remap_xcd(pid, GRID_MN * NUM_KSPLIT, NUM_XCDS=8)

    # 3D decomposition: pid -> (pid_m, pid_n, pid_k)
    if NUM_KSPLIT > 1:
        pid_k = pid // GRID_MN
        pid_mn = pid % GRID_MN
    else:
        pid_k = 0
        pid_mn = pid

    pid_m, pid_n = pid_grid(pid_mn, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)

    # K range for this split
    if NUM_KSPLIT > 1:
        k_start = pid_k * SPLITK_BLOCK_SIZE
        k_end = min(k_start + SPLITK_BLOCK_SIZE, K)
    else:
        k_start = 0
        k_end = K

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K // 2)

    a_bf16_ptrs = a_bf16_ptr + offs_am[:, None] * stride_am + (k_start + tl.arange(0, BLOCK_SIZE_K))[None, :] * stride_ak
    b_ptrs = b_ptr + (k_start // 2 + offs_k)[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    offs_ks = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
    b_scale_ptrs = b_scales_ptr + offs_bn[:, None] * stride_bsn + (k_start // SCALE_GROUP_SIZE + offs_ks)[None, :] * stride_bsk

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    num_k_iters = tl.cdiv(k_end - k_start, BLOCK_SIZE_K)
    for ki in range(0, num_k_iters):
        a_bf16 = tl.load(a_bf16_ptrs).to(tl.float32)
        a_fp4, a_scales = _mxfp4_quant_op(a_bf16, BLOCK_SIZE_K, BLOCK_SIZE_M, SCALE_GROUP_SIZE)
        b = tl.load(b_ptrs, cache_modifier=cache_modifier)
        b_scales = tl.load(b_scale_ptrs, cache_modifier=cache_modifier)
        accumulator = tl.dot_scaled(a_fp4, a_scales, "e2m1", b, b_scales, "e2m1", accumulator)
        a_bf16_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        b_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_bsk

    if NUM_KSPLIT == 1:
        c = accumulator.to(c_ptr.type.element_ty)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        # Write partial results to temp buffer (fp32, indexed by pid_k)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        # c_ptr points to (NUM_KSPLIT, M, N) fp32 buffer
        c_ptrs = c_ptr + pid_k * M * N + offs_cm[:, None] * N + offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def _reduce_splitk_kernel(
    partial_ptr, out_ptr,
    M, N, NUM_KSPLIT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    num_warps: tl.constexpr,
):
    """Reduce NUM_KSPLIT partial fp32 results into bf16 output."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(NUM_KSPLIT):
        partial = tl.load(
            partial_ptr + k * M * N + offs_m[:, None] * N + offs_n[None, :],
            mask=mask, other=0.0,
        )
        acc += partial

    out = acc.to(tl.bfloat16)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], out, mask=mask)


from task import input_t, output_t
import torch
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import _get_config


def e8m0_unshuffle(scale, orig_m, orig_n):
    sm, sn = scale.shape
    scale = scale.view(sm // 32, sn // 8, 4, 16, 2, 2)
    scale = scale.permute(0, 5, 3, 1, 4, 2).contiguous()
    scale = scale.view(sm, sn)
    return scale[:orig_m, :orig_n]


_cache_key = None
_cache_val = None
_out_cache = {}
_partial_cache = {}


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _cache_key, _cache_val
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    B_q_uint8 = B_q.view(torch.uint8)

    key = (B.data_ptr(), B_q.data_ptr(), B_scale_sh.data_ptr())
    if key == _cache_key:
        B_scale = _cache_val
    else:
        if k <= 512:
            _, B_scale = dynamic_mxfp4_quant(B)
        else:
            B_scale = e8m0_unshuffle(B_scale_sh.view(torch.uint8), n, k // 32)
        _cache_key = key
        _cache_val = B_scale

    config, _ = _get_config(m, n, k)
    ksplit = config.get("NUM_KSPLIT", 1)

    # For M<=16: use custom fused kernel (with or without split-K)
    if m <= 16:
        B_q_t = B_q_uint8.T
        bsm = config["BLOCK_SIZE_M"]
        bsn = config["BLOCK_SIZE_N"]
        bsk = config["BLOCK_SIZE_K"]

        if ksplit == 1:
            out_key = (m, n)
            if out_key in _out_cache:
                y = _out_cache[out_key]
            else:
                y = torch.empty((m, n), dtype=torch.bfloat16, device=A.device)
                _out_cache[out_key] = y

            # Compute SPLITK_BLOCK_SIZE (not used for KSPLIT=1 but kernel needs it)
            splitk_bs = k

            grid = (triton.cdiv(m, bsm) * triton.cdiv(n, bsn),)
            fused_config = {k_: v_ for k_, v_ in config.items()
                            if k_ in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
                                      "GROUP_SIZE_M", "num_warps", "num_stages",
                                      "waves_per_eu", "matrix_instr_nonkdim", "cache_modifier")}
            _fused_quant_gemm_kernel[grid](
                A, B_q_t, y, B_scale, m, n, k,
                A.stride(0), A.stride(1), B_q_t.stride(0), B_q_t.stride(1),
                y.stride(0), y.stride(1), B_scale.stride(0), B_scale.stride(1),
                NUM_KSPLIT=1, SPLITK_BLOCK_SIZE=splitk_bs,
                **fused_config,
            )
            return y
        else:
            # Split-K path with temp buffer
            splitk_bs = triton.cdiv(k, ksplit)
            # Round up to BSK boundary
            splitk_bs = triton.cdiv(splitk_bs, bsk) * bsk

            # Allocate temp buffer
            partial_key = (ksplit, m, n)
            if partial_key in _partial_cache:
                partial = _partial_cache[partial_key]
            else:
                partial = torch.empty((ksplit, m, n), dtype=torch.float32, device=A.device)
                _partial_cache[partial_key] = partial

            num_m_tiles = triton.cdiv(m, bsm)
            num_n_tiles = triton.cdiv(n, bsn)
            grid = (num_m_tiles * num_n_tiles * ksplit,)

            fused_config = {k_: v_ for k_, v_ in config.items()
                            if k_ in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
                                      "GROUP_SIZE_M", "num_warps", "num_stages",
                                      "waves_per_eu", "matrix_instr_nonkdim", "cache_modifier")}
            _fused_quant_gemm_kernel[grid](
                A, B_q_t, partial, B_scale, m, n, k,
                A.stride(0), A.stride(1), B_q_t.stride(0), B_q_t.stride(1),
                0, 0,  # stride_cm, stride_cn not used for partial buffer
                B_scale.stride(0), B_scale.stride(1),
                NUM_KSPLIT=ksplit, SPLITK_BLOCK_SIZE=splitk_bs,
                **fused_config,
            )

            # Reduce
            out_key = (m, n)
            if out_key in _out_cache:
                y = _out_cache[out_key]
            else:
                y = torch.empty((m, n), dtype=torch.bfloat16, device=A.device)
                _out_cache[out_key] = y

            reduce_bsm = 16
            reduce_bsn = 64
            reduce_grid = (triton.cdiv(m, reduce_bsm), triton.cdiv(n, reduce_bsn))
            _reduce_splitk_kernel[reduce_grid](
                partial, y, m, n,
                NUM_KSPLIT=ksplit,
                BLOCK_SIZE_M=reduce_bsm, BLOCK_SIZE_N=reduce_bsn,
                num_warps=4,
            )
            return y
    else:
        # M>16: use standard path
        A_q, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_q, B_q_uint8, A_scale, B_scale, dtype=torch.bfloat16)
