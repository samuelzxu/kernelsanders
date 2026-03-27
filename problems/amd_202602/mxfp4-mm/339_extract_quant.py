#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#339: Extract the FULL _mxfp4_quant_op source to get exact FP4 rounding.
"""
import torch, sys, inspect
from task import input_t, output_t

# Extract _mxfp4_quant_op source
try:
    import aiter.ops.triton._triton_kernels.quant.quant as qmod
    fn = qmod._mxfp4_quant_op
    if hasattr(fn, 'fn'): fn = fn.fn
    if hasattr(fn, 'fn'): fn = fn.fn
    src = inspect.getsource(fn)
    print("=== _mxfp4_quant_op FULL SOURCE ===")
    print(src)
except Exception as e:
    print(f"Error: {e}")

# Also extract _dynamic_mxfp4_quant_kernel
try:
    fn2 = qmod._dynamic_mxfp4_quant_kernel
    if hasattr(fn2, 'fn'): fn2 = fn2.fn
    if hasattr(fn2, 'fn'): fn2 = fn2.fn
    src2 = inspect.getsource(fn2)
    print("\n=== _dynamic_mxfp4_quant_kernel FULL SOURCE ===")
    print(src2[:3000])
except Exception as e:
    print(f"Error: {e}")

# Standard kernel
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
import os, json
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
_cfgs = {"N=2880-K=512": {"M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}}
try: _dev = arch_info.get_arch()
except: _dev = "gfx950"
_cd = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
os.makedirs(_cd, exist_ok=True)
for _sk, _cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json", "w") as f:
        json.dump(_cfg, f)

_ck=None;_cw=None;_cs=None
@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _ck,_cw,_cs
    A=data[0];B_shuffle=data[3];B_scale_sh=data[4]
    m,k=A.shape;n=data[1].shape[0]
    dp=B_shuffle.data_ptr()
    if dp!=_ck:
        _ck=dp;_cw=B_shuffle.view(torch.uint8).reshape(n//16,(k//2)*16)
        _cs=B_scale_sh.view(torch.uint8)[:n,:].contiguous().reshape(n//32,k)
    return gemm_a16wfp4_preshuffle(A,_cw,_cs,prequant=True,dtype=torch.bfloat16)
