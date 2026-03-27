#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#469: Test if gemm_a4w4 needs shuffled A scales or raw scales.
"""
import torch
from task import input_t, output_t
import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle

# Warmup
for _m,_n,_k in [(256,3072,1536)]:
    _A=torch.randn(_m,_k,dtype=torch.bfloat16,device="cuda")
    _aq,_as=dynamic_mxfp4_quant(_A)
    _aq_t=_aq.view(dtypes.fp4x2)
    _as_sh=e8m0_shuffle(_as).view(dtypes.fp8_e8m0)
    _as_raw=_as.view(dtypes.fp8_e8m0)
    from aiter.ops.shuffle import shuffle_weight
    _bq=torch.zeros(_n,_k//2,dtype=torch.uint8,device="cuda").view(dtypes.fp4x2)
    _bsh=shuffle_weight(_bq,layout=(16,16))
    _bsc=torch.zeros(_n,_k//32,dtype=torch.uint8,device="cuda").view(dtypes.fp8_e8m0)
    # Test shuffled
    out_sh=aiter.gemm_a4w4(_aq_t,_bsh,_as_sh,_bsc,dtype=dtypes.bf16,bpreshuffle=True)
    # Test raw (no shuffle)
    out_raw=aiter.gemm_a4w4(_aq_t,_bsh,_as_raw,_bsc,dtype=dtypes.bf16,bpreshuffle=True)
    diff=torch.abs(out_sh.float()-out_raw.float()).max().item()
    print(f"Max diff shuffled vs raw: {diff}")
    print(f"Out shuffled sample: {out_sh[0,:5]}")
    print(f"Out raw sample: {out_raw[0,:5]}")
torch.cuda.empty_cache()

# Use preshuffle as baseline
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
import os, json
_cfgs={"N=3072-K=1536":{"M_LEQ_256":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":32,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)

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
