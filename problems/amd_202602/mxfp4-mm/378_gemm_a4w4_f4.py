#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#378: Benchmark gemm_a4w4 with float4_e2m1fn_x2 dtype for K=1536 M=256.
gemm_a4w4 dispatches to blockscale when dtype is float4.
Use cached B + hiprtc quant for A.
"""
import torch, os, json, time
from task import input_t, output_t
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter import gemm_a4w4
import aiter

e8m0_sh = aiter.fp4_utils.e8m0_shuffle

# Warmup gemm_a4w4 with float4 dtype for K=1536
M,N,K = 256,3072,1536
_A = torch.randn(M,K,dtype=torch.bfloat16,device="cuda")
_Aq, _As = dynamic_mxfp4_quant(_A)
_As = _As.contiguous()
_As_sh = e8m0_sh(_As)
_Aq_f4 = _Aq.view(torch.float4_e2m1fn_x2)
_Bq = torch.zeros(N,K//2,dtype=torch.uint8,device="cuda")
_Bq_f4 = _Bq.view(torch.float4_e2m1fn_x2)
_Bs = torch.zeros(N,K//32,dtype=torch.uint8,device="cuda")

print("=== Warmup ===")
t0 = time.time()
for _ in range(3):
    _ = gemm_a4w4(_Aq_f4, _Bq_f4, _As_sh, _Bs, bpreshuffle=True)
torch.cuda.synchronize()
print(f"Warmup: {time.time()-t0:.2f}s")

# Benchmark GEMM-only
print("=== GEMM timing ===")
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100):
    _ = gemm_a4w4(_Aq_f4, _Bq_f4, _As_sh, _Bs, bpreshuffle=True)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / 100
print(f"gemm_a4w4 float4: {ms*1000:.1f}µs")

# Also time dynamic_mxfp4_quant + e8m0_shuffle
start.record()
for _ in range(100):
    _q, _s = dynamic_mxfp4_quant(_A)
    _s_sh = e8m0_sh(_s.contiguous())
end.record()
torch.cuda.synchronize()
ms2 = start.elapsed_time(end) / 100
print(f"quant+shuffle: {ms2*1000:.1f}µs")

del _A, _Aq, _As, _Bq, _Bs, _Aq_f4, _Bq_f4, _As_sh
torch.cuda.empty_cache()

# Standard preshuffle kernel for all shapes
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
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
