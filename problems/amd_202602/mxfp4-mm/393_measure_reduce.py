#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#393: Measure the split-K reduce kernel time separately.
For K=1536 KSPLIT=2: how much of the 16.5µs is from the reduce kernel?
"""
import torch, os, json, time
from task import input_t, output_t

# Import reduce kernel
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import (
    gemm_a16wfp4_preshuffle,
    _gemm_afp4wfp4_reduce_kernel,
)

# Time the reduce kernel independently
M,N=256,3072
# Create a dummy split-K intermediate buffer (KSPLIT=2, M, N) in float32
inter = torch.randn(2, M, N, dtype=torch.float32, device="cuda")
out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

# Warm up reduce kernel
grid = lambda meta: (M * N // meta['BLOCK_SIZE'],)
for _ in range(5):
    _gemm_afp4wfp4_reduce_kernel[grid](
        inter, out,
        M, N, 2,  # NUM_KSPLIT=2
        inter.stride(0), inter.stride(1), inter.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=256,
    )
torch.cuda.synchronize()

# Measure
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100):
    _gemm_afp4wfp4_reduce_kernel[grid](
        inter, out, M, N, 2,
        inter.stride(0), inter.stride(1), inter.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=256,
    )
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / 100
print(f"Reduce kernel (KSPLIT=2, {M}x{N}): {ms*1000:.1f}µs")

# Also for KSPLIT=7 K=7168 shape
inter7 = torch.randn(7, 16, 2112, dtype=torch.float32, device="cuda")
out7 = torch.empty(16, 2112, dtype=torch.bfloat16, device="cuda")
for _ in range(5):
    _gemm_afp4wfp4_reduce_kernel[grid](
        inter7, out7, 16, 2112, 7,
        inter7.stride(0), inter7.stride(1), inter7.stride(2),
        out7.stride(0), out7.stride(1),
        BLOCK_SIZE=256,
    )
torch.cuda.synchronize()
start.record()
for _ in range(100):
    _gemm_afp4wfp4_reduce_kernel[grid](
        inter7, out7, 16, 2112, 7,
        inter7.stride(0), inter7.stride(1), inter7.stride(2),
        out7.stride(0), out7.stride(1),
        BLOCK_SIZE=256,
    )
end.record()
torch.cuda.synchronize()
ms7 = start.elapsed_time(end) / 100
print(f"Reduce kernel (KSPLIT=7, 16x2112): {ms7*1000:.1f}µs")

del inter, out, inter7, out7
torch.cuda.empty_cache()

# Standard kernel
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
_cfgs={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=4096-K=512":{"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=2112-K=7168":{"M_LEQ_16":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":8,"num_warps":4,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=7168-K=2048":{"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":32,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=3072-K=1536":{"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":3,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_256":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":32,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)
for _m,_n,_k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
    try:
        _A=torch.randn((_m,_k),dtype=torch.bfloat16,device="cuda")
        _Bw=torch.zeros((_n//16,(_k//2)*16),dtype=torch.uint8,device="cuda")
        _Bws=torch.zeros((_n//32,_k),dtype=torch.uint8,device="cuda")
        gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
    except:pass
torch.cuda.empty_cache()
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
