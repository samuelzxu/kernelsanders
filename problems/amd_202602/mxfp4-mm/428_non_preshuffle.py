#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#428: Test gemm_a16wfp4 (non-preshuffle) with raw B_q and B_scale.
Different internal B loading pattern might be faster for some shapes.
"""
import torch, os, json, inspect
from task import input_t, output_t
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4, gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info

# Check gemm_a16wfp4 signature
print("=== gemm_a16wfp4 signature ===")
try:
    src = inspect.getsource(gemm_a16wfp4)
    print(src[:1000])
except: print("can't get source")

# Test it
_cfgs={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
# Write non-preshuffle config
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4-{_sk}.json","w") as f:json.dump(_cfg,f)
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)

M,N,K=4,2880,512
A=torch.randn(M,K,dtype=torch.bfloat16,device="cuda")
B_q=torch.zeros(N,K//2,dtype=torch.uint8,device="cuda")
B_scale=torch.zeros(N,K//32,dtype=torch.uint8,device="cuda")

# Test non-preshuffle
try:
    r = gemm_a16wfp4(A, B_q, B_scale, prequant=True, dtype=torch.bfloat16)
    print(f"gemm_a16wfp4 works: {r.shape}")
except Exception as e:
    print(f"gemm_a16wfp4 error: {str(e)[:200]}")

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
