#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4-MM #243: Diagnostic - check A_scale and B_scale shapes/formats from dynamic_mxfp4_quant.
"""
import os, sys, torch
from task import input_t, output_t
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.quant import dynamic_mxfp4_quant
import json

_CONFIGS = {
    "N=2880-K=512": {"M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}},
    "N=4096-K=512": {"M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}},
    "N=7168-K=2048": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}},
    "N=3072-K=1536": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 3, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}},
}
def _inject():
    try: dev = arch_info.get_arch()
    except: dev = "gfx950"
    cd = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
    os.makedirs(cd, exist_ok=True)
    for sk, cfg in _CONFIGS.items():
        with open(f"{cd}/{dev}-GEMM-A16WFP4_PRESHUFFLED-{sk}.json", "w") as f:
            json.dump(cfg, f)
try: _inject()
except: pass

_bc = [None, None, None]
_first_call = [True]

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]

    if _first_call[0]:
        _first_call[0] = False
        # Diagnostic: check shapes and formats
        A_q, A_scale = dynamic_mxfp4_quant(A)
        print(f"[243] A.shape={A.shape} dtype={A.dtype}", file=sys.stderr)
        print(f"[243] A_q.shape={A_q.shape} dtype={A_q.dtype}", file=sys.stderr)
        print(f"[243] A_scale.shape={A_scale.shape} dtype={A_scale.dtype}", file=sys.stderr)
        print(f"[243] A_scale uint8 view shape={A_scale.view(torch.uint8).shape}", file=sys.stderr)
        print(f"[243] B_q.shape={B_q.shape} dtype={B_q.dtype}", file=sys.stderr)
        print(f"[243] B_scale_sh.shape={B_scale_sh.shape} dtype={B_scale_sh.dtype}", file=sys.stderr)

        # Check if A_scale is padded
        a_sc = A_scale.view(torch.uint8)
        print(f"[243] A_scale[0,:5] = {a_sc[0,:5].tolist()}", file=sys.stderr)
        print(f"[243] A_scale[-1,:5] = {a_sc[-1,:5].tolist()}", file=sys.stderr)

        # Check B_scale_sh
        b_sc = B_scale_sh.view(torch.uint8)
        print(f"[243] B_scale_sh[0,:5] = {b_sc[0,:5].tolist()}", file=sys.stderr)
        print(f"[243] B_scale_sh shape={b_sc.shape}", file=sys.stderr)

        # Check if A_scale stride is what we expect
        print(f"[243] A_scale strides={A_scale.stride()}", file=sys.stderr)
        print(f"[243] A_scale uint8 strides={a_sc.stride()}", file=sys.stderr)

        # Check if A_q packed FP4 matches B_q format
        print(f"[243] A_q[0,:5] uint8 = {A_q.view(torch.uint8)[0,:5].tolist()}", file=sys.stderr)
        print(f"[243] B_q[0,:5] uint8 = {B_q.view(torch.uint8)[0,:5].tolist()}", file=sys.stderr)

    # Use preshuffle (known correct)
    key = (B_shuffle.data_ptr(), B_scale_sh.data_ptr())
    if key != _bc[0]:
        _bc[0] = key
        _bc[1] = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        _bc[2] = B_scale_sh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
    return gemm_a16wfp4_preshuffle(A, _bc[1], _bc[2], prequant=True, dtype=torch.bfloat16)
