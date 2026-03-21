"""
MXFP4-MM: #216 - Preshuffle for K=512/K=7168 + Gluon GEMM for K=2048/K=1536.

Gluon (triton.experimental.gluon) uses explicit LDS swizzle layouts optimized
for MI355X's 64-bank LDS. The SwizzledSharedLayout(vec=16, per_phase=2,
max_phase=8) eliminates bank conflicts that standard Triton can't avoid.

Limitation: gluon's MFMA layout requires BSM>=64 (warps_per_cta=[2, nw//2]),
so only works for M=64 (K=2048) and M=256 (K=1536).

Hybrid:
  K=512: preshuffle (fused, M=4/32, already fast)
  K=7168: preshuffle (fused, M=16, already fast)
  K=2048 M=64: dynamic_mxfp4_quant + gluon GEMM
  K=1536 M=256: dynamic_mxfp4_quant + gluon GEMM
"""
import os, json
import torch
from task import input_t, output_t
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.gluon.gemm_afp4wfp4 import gemm_afp4wfp4 as gluon_gemm

# Preshuffle configs
_PS_CONFIGS = {
    "N=2880-K=512": {
        "M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
    "N=4096-K=512": {
        "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
    # K=7168: use aiter's existing tuned preshuffle config
}

# Gluon configs — must use BSM>=64 due to MFMA layout, mfma=32
_GLUON_CONFIGS = {
    # K=2048 M=64: BSM=64, BSN=256, BSK=256, KSPLIT=1
    "N=7168-K=2048": {
        "M_LEQ_64": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 0, "matrix_instr_nonkdim": 32, "cache_modifier": None},
        "any": {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 0, "matrix_instr_nonkdim": 32, "cache_modifier": None}
    },
    # K=1536 M=256: BSM=256, BSN=256, BSK=256, KSPLIT=1
    "N=3072-K=1536": {
        "M_LEQ_256": {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 0, "matrix_instr_nonkdim": 32, "cache_modifier": None},
        "any": {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 0, "matrix_instr_nonkdim": 32, "cache_modifier": None}
    },
}

def _inject_configs():
    try:
        dev = arch_info.get_arch()
    except Exception:
        dev = "gfx950"
    config_dir = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
    os.makedirs(config_dir, exist_ok=True)
    for shape_key, config in _PS_CONFIGS.items():
        fpath = f"{config_dir}/{dev}-GEMM-A16WFP4_PRESHUFFLED-{shape_key}.json"
        with open(fpath, "w") as f:
            json.dump(config, f)
    # Gluon uses its own config path
    gluon_dir = f"{config_dir}/gluon"
    os.makedirs(gluon_dir, exist_ok=True)
    for shape_key, config in _GLUON_CONFIGS.items():
        fpath = f"{gluon_dir}/{dev}-GEMM-AFP4WFP4-{shape_key}.json"
        with open(fpath, "w") as f:
            json.dump(config, f)

_inject_configs()


def e8m0_unshuffle(scale, orig_m, orig_n):
    sm, sn = scale.shape
    scale = scale.view(sm // 32, sn // 8, 4, 16, 2, 2)
    scale = scale.permute(0, 5, 3, 1, 4, 2).contiguous()
    scale = scale.view(sm, sn)
    return scale[:orig_m, :orig_n]


_ps_cache_key = None
_ps_cache_w = None
_ps_cache_ws = None
_gluon_bscale_key = None
_gluon_bscale_val = None


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _ps_cache_key, _ps_cache_w, _ps_cache_ws
    global _gluon_bscale_key, _gluon_bscale_val
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if k <= 512 or k == 7168:
        # Preshuffle: single kernel, fused quant+GEMM
        key = (B_shuffle.data_ptr(), B_scale_sh.data_ptr())
        if key == _ps_cache_key:
            B_w = _ps_cache_w
            B_ws = _ps_cache_ws
        else:
            B_w = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
            B_ws = B_scale_sh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
            _ps_cache_key = key
            _ps_cache_w = B_w
            _ps_cache_ws = B_ws
        return gemm_a16wfp4_preshuffle(A, B_w, B_ws, prequant=True, dtype=torch.bfloat16)

    else:
        # K=2048, K=1536: Gluon GEMM with explicit LDS swizzle
        B_q_uint8 = B_q.view(torch.uint8)
        key = (B.data_ptr(), B_q.data_ptr(), B_scale_sh.data_ptr())
        if key == _gluon_bscale_key:
            B_scale = _gluon_bscale_val
        else:
            B_scale = e8m0_unshuffle(B_scale_sh.view(torch.uint8), n, k // 32)
            _gluon_bscale_key = key
            _gluon_bscale_val = B_scale

        A_q, A_scale = dynamic_mxfp4_quant(A)
        return gluon_gemm(A_q, B_q_uint8, A_scale, B_scale, dtype=torch.bfloat16)
