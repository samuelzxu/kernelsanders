#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#400: Benchmark gemm_afp4wfp4_preshuffle vs gemm_a16wfp4_preshuffle for K=1536 M=256.
afp4wfp4 takes pre-quantized A — if the GEMM kernel itself is faster (no inline quant),
we might beat a16wfp4 even with separate quant overhead.
B is pre-quantized+preshuffled → quant B during warmup.
A quant done by the preshuffle kernel inline. But for afp4wfp4: need separate A quant.
Key insight: afp4wfp4 GEMM might be faster because it doesn't do inline quant!
"""
import torch, os, json, time
from task import input_t, output_t
from aiter.ops.triton.quant import dynamic_mxfp4_quant
import aiter.ops.triton.gemm.basic.gemm_afp4wfp4 as afmod
import aiter

e8m0_sh = aiter.fp4_utils.e8m0_shuffle

# Time afp4wfp4_preshuffle GEMM alone
print("=== gemm_afp4wfp4_preshuffle timing ===")

from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info

# Config for AFP4WFP4 (note: different config file naming!)
# The config uses "GEMM-AFP4WFP4" prefix
_cfgs_afp4 = {
    "N=3072-K=1536": {
        "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256,
                      "GROUP_SIZE_M": 4, "NUM_KSPLIT": 2, "num_warps": 8,
                      "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32,
                      "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256,
                "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8,
                "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16,
                "cache_modifier": None}
    }
}
try: _dev = arch_info.get_arch()
except: _dev = "gfx950"
_cd = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
os.makedirs(_cd, exist_ok=True)
# Write AFP4WFP4 configs
for _sk, _cfg in _cfgs_afp4.items():
    with open(f"{_cd}/{_dev}-GEMM-AFP4WFP4-{_sk}.json", "w") as f:
        json.dump(_cfg, f)
    # Also preshuffle variant
    with open(f"{_cd}/{_dev}-GEMM-AFP4WFP4_PRESHUFFLED-{_sk}.json", "w") as f:
        json.dump(_cfg, f)

M,N,K=256,3072,1536
A=torch.randn(M,K,dtype=torch.bfloat16,device="cuda")
A_q,A_s=dynamic_mxfp4_quant(A)
A_s_c=A_s.contiguous()
A_s_sh=e8m0_sh(A_s_c)
B_q=torch.zeros(N,K//2,dtype=torch.uint8,device="cuda")
B_w=B_q.view(torch.uint8).reshape(N//16,(K//2)*16)
B_scale=torch.zeros(N//32,K,dtype=torch.uint8,device="cuda")

# Warmup
for _ in range(5):
    afmod.gemm_afp4wfp4_preshuffle(A_q, B_w, A_s_sh, B_scale, dtype=torch.bfloat16)
torch.cuda.synchronize()

# Time GEMM only (no quant)
start=torch.cuda.Event(enable_timing=True)
end=torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100):
    afmod.gemm_afp4wfp4_preshuffle(A_q, B_w, A_s_sh, B_scale, dtype=torch.bfloat16)
end.record()
torch.cuda.synchronize()
ms=start.elapsed_time(end)/100
print(f"afp4wfp4_preshuffle GEMM: {ms*1000:.1f}us")

# Time quant + shuffle
start.record()
for _ in range(100):
    _q,_s = dynamic_mxfp4_quant(A)
    _s_sh = e8m0_sh(_s.contiguous())
end.record()
torch.cuda.synchronize()
ms2=start.elapsed_time(end)/100
print(f"dynamic_mxfp4_quant+shuffle: {ms2*1000:.1f}us")

# For reference: time a16wfp4_preshuffle (fused)
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
_cfgs_a16={"N=3072-K=1536":{"M_LEQ_256":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":32,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
for _sk,_cfg in _cfgs_a16.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)
_cw=B_q.view(torch.uint8).reshape(N//16,(K//2)*16)
_cs=torch.zeros(N//32,K,dtype=torch.uint8,device="cuda")
for _ in range(5):
    gemm_a16wfp4_preshuffle(A,_cw,_cs,prequant=True,dtype=torch.bfloat16)
torch.cuda.synchronize()
start.record()
for _ in range(100):
    gemm_a16wfp4_preshuffle(A,_cw,_cs,prequant=True,dtype=torch.bfloat16)
end.record()
torch.cuda.synchronize()
ms3=start.elapsed_time(end)/100
print(f"a16wfp4_preshuffle (fused): {ms3*1000:.1f}us")

del A,A_q,A_s,A_s_c,A_s_sh,B_q,B_w,B_scale;torch.cuda.empty_cache()

# Standard submission
_cfgs2={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=4096-K=512":{"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=2112-K=7168":{"M_LEQ_16":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":8,"num_warps":4,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=7168-K=2048":{"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":32,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
for _sk,_cfg in _cfgs2.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)
for _m,_n,_k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048)]:
    try:
        _A=torch.randn((_m,_k),dtype=torch.bfloat16,device="cuda")
        _Bw=torch.zeros((_n//16,(_k//2)*16),dtype=torch.uint8,device="cuda")
        _Bws=torch.zeros((_n//32,_k),dtype=torch.uint8,device="cuda")
        gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
    except:pass
torch.cuda.empty_cache()
_ck=None;_cw2=None;_cs2=None
@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _ck,_cw2,_cs2
    A=data[0];B_shuffle=data[3];B_scale_sh=data[4]
    m,k=A.shape;n=data[1].shape[0]
    dp=B_shuffle.data_ptr()
    if dp!=_ck:
        _ck=dp;_cw2=B_shuffle.view(torch.uint8).reshape(n//16,(k//2)*16)
        _cs2=B_scale_sh.view(torch.uint8)[:n,:].contiguous().reshape(n//32,k)
    return gemm_a16wfp4_preshuffle(A,_cw2,_cs2,prequant=True,dtype=torch.bfloat16)
