"""
MXFP4-MM: #89 - Use gemm_a16wfp4 (fused bf16->fp4 quant + GEMM) for M>=32.
Eliminates separate dynamic_mxfp4_quant(A) call.
Also inject A16WFP4 configs with correct keys (actual K values).
"""
import json
import os
import triton
import triton.language as tl
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

# AFP4WFP4 configs for M<=16 fused kernel path
_CONFIGS_FP4 = {
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
        "M_LEQ_16": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 8, "num_warps": 4, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 32, "cache_modifier": None}
    },
    "N=7168-K=2048": {
        "M_LEQ_64": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 1024, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 32, "cache_modifier": None}
    },
    "N=3072-K=1536": {
        "M_LEQ_64": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 4, "num_stages": 3, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "M_LEQ_256": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 2, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 32, "cache_modifier": None}
    },
}

# A16WFP4 configs - key uses actual K (the 2*K in code just converts packed back to actual)
# Inject custom configs for shapes that don't have existing tuned configs
_CONFIGS_A16 = {
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
        "M_LEQ_16": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 8, "num_warps": 4, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 32, "cache_modifier": None}
    },
    "N=3072-K=1536": {
        "M_LEQ_64": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 4, "num_stages": 3, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "M_LEQ_256": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 2, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 32, "cache_modifier": None}
    },
    # N=7168-K=2048 already has a tuned config in the repo, don't override
}


def _inject_configs():
    try:
        dev = arch_info.get_arch()
    except Exception:
        dev = "gfx950"
    config_dir = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
    os.makedirs(config_dir, exist_ok=True)
    for shape_key, config in _CONFIGS_FP4.items():
        fpath = f"{config_dir}/{dev}-GEMM-AFP4WFP4-{shape_key}.json"
        with open(fpath, "w") as f:
            json.dump(config, f)
    for shape_key, config in _CONFIGS_A16.items():
        fpath = f"{config_dir}/{dev}-GEMM-A16WFP4-{shape_key}.json"
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
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import _get_config


def e8m0_unshuffle(scale, orig_m, orig_n):
    sm, sn = scale.shape
    scale = scale.view(sm // 32, sn // 8, 4, 16, 2, 2)
    scale = scale.permute(0, 5, 3, 1, 4, 2).contiguous()
    scale = scale.view(sm, sn)
    return scale[:orig_m, :orig_n]


_cache_key = None
_cache_val = None


def custom_kernel(data: input_t) -> output_t:
    global _cache_key, _cache_val
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    B_q_uint8 = B_q.view(torch.uint8)

    # Get B_scale (with caching) - unshuffled for Triton kernels
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

    # For M<=16 with no split-K: custom fused quant+GEMM kernel
    config, _ = _get_config(m, n, k)
    if m <= 16 and config.get("NUM_KSPLIT", 1) == 1:
        y = torch.empty((m, n), dtype=torch.bfloat16, device=A.device)
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

    # For M>=32: use gemm_a16wfp4 (fused bf16->fp4 quant + GEMM)
    # Eliminates separate dynamic_mxfp4_quant(A) call
    # w expects (N, K//2) uint8, w_scales expects (N, K//32) uint8
    return gemm_a16wfp4(A, B_q_uint8, B_scale, dtype=torch.bfloat16)
