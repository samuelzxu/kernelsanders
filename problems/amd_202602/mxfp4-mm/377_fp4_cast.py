#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#377: Try gemm_a4w4_asm with float4_e2m1fn_x2 dtype.
"""
import torch, os, json, time
from task import input_t, output_t
from aiter.ops.triton.quant import dynamic_mxfp4_quant

# Test gemm_a4w4_asm with float4 dtype
print("=== FP4 dtype test ===")
import aiter
from aiter import gemm_a4w4_asm

M, N, K = 256, 3072, 1536
A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
A_q, A_scale = dynamic_mxfp4_quant(A)
A_scale = A_scale.contiguous()

# View A_q as float4_e2m1fn_x2
A_q_f4 = A_q.view(torch.float4_e2m1fn_x2)
print(f"A_q: {A_q.shape} {A_q.dtype}")
print(f"A_q_f4: {A_q_f4.shape} {A_q_f4.dtype}")

# Create B in preshuffle format
B_q = torch.zeros(N, K//2, dtype=torch.uint8, device="cuda")
B_q_f4 = B_q.view(torch.float4_e2m1fn_x2)
B_scale = torch.zeros(N, K//32, dtype=torch.uint8, device="cuda")
out = torch.empty(((M+31)//32*32, N), dtype=torch.bfloat16, device="cuda")

# Also try e8m0_shuffle on A_scale
e8m0_sh = aiter.fp4_utils.e8m0_shuffle
A_scale_sh = e8m0_sh(A_scale)
print(f"A_scale: {A_scale.shape}")
print(f"A_scale_sh: {A_scale_sh.shape}")

# Try various calls
for desc, a_arg, b_arg, as_arg, bs_arg in [
    ("f4,f4,raw,raw", A_q_f4, B_q_f4, A_scale, B_scale),
    ("f4,u8,raw,raw", A_q_f4, B_q, A_scale, B_scale),
    ("u8,f4,raw,raw", A_q, B_q_f4, A_scale, B_scale),
    ("f4,f4,sh,raw", A_q_f4, B_q_f4, A_scale_sh, B_scale),
]:
    try:
        r = gemm_a4w4_asm(a_arg, b_arg, as_arg, bs_arg, out,
                          "f4gemm_bf16_per1x32Fp4_BpreShuffle_192x128",
                          bpreshuffle=True)
        print(f"  {desc}: OK! shape={r.shape}")
        break
    except Exception as e:
        print(f"  {desc}: {str(e)[:100]}")

# Try gemm_a4w4 (high-level) with float4 dtype
from aiter import gemm_a4w4
for desc, a_arg, as_arg in [
    ("f4,raw", A_q_f4, A_scale),
    ("f4,sh", A_q_f4, A_scale_sh),
]:
    try:
        r = gemm_a4w4(a_arg, B_q_f4, as_arg, B_scale, bpreshuffle=True)
        print(f"  gemm_a4w4 {desc}: OK! shape={r.shape}")
    except Exception as e:
        print(f"  gemm_a4w4 {desc}: {str(e)[:100]}")

# Standard kernel
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
