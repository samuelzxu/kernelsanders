#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#381: Check gemm_a4w4 float4 correctness against reference for K=1536 M=256.
"""
import torch, os, json
from task import input_t, output_t
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter import gemm_a4w4
from reference import generate_input

# Generate actual test data
data = generate_input(m=256, n=3072, k=1536, seed=7856)
A, B, B_q, B_shuffle, B_scale_sh = data
m,k=A.shape;n=B.shape[0]

# Reference output via preshuffle
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
_cfgs={"N=3072-K=1536":{"M_LEQ_256":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":32,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)
_cw=B_shuffle.view(torch.uint8).reshape(n//16,(k//2)*16)
_cs=B_scale_sh.view(torch.uint8)[:n,:].contiguous().reshape(n//32,k)
ref=gemm_a16wfp4_preshuffle(A,_cw,_cs,prequant=True,dtype=torch.bfloat16)

# gemm_a4w4 float4 with raw B_q and unshuffled B_scale
A_q, A_s = dynamic_mxfp4_quant(A)
A_s = A_s.contiguous()
A_q_f4 = A_q.view(torch.float4_e2m1fn_x2)
B_q_f4 = B_q.view(torch.float4_e2m1fn_x2)

# Unshuffle B_scale
sm,sn=B_scale_sh.shape
t=B_scale_sh.view(sm//32,sn//8,4,16,2,2).permute(0,5,3,1,4,2).contiguous().view(sm,sn)[:n,:k//32].contiguous()

# Try with unshuffled B_scale
out1=gemm_a4w4(A_q_f4,B_q_f4,A_s,t,bpreshuffle=True)
diff1=(ref-out1).abs()
print(f"Unshuffled B_scale: max_err={diff1.max().item():.1f}, mean_err={diff1.mean().item():.3f}")

# Try with shuffled B_scale
bs_sh=B_scale_sh[:n,:].contiguous()
out2=gemm_a4w4(A_q_f4,B_q_f4,A_s,bs_sh,bpreshuffle=True)
diff2=(ref-out2).abs()
print(f"Shuffled B_scale: max_err={diff2.max().item():.1f}, mean_err={diff2.mean().item():.3f}")

# Try with shuffled A_scale
import aiter
A_s_sh=aiter.fp4_utils.e8m0_shuffle(A_s)
out3=gemm_a4w4(A_q_f4,B_q_f4,A_s_sh,t,bpreshuffle=True)
diff3=(ref-out3).abs()
print(f"Shuffled A + unshuffled B: max_err={diff3.max().item():.1f}, mean_err={diff3.mean().item():.3f}")

# Both shuffled
out4=gemm_a4w4(A_q_f4,B_q_f4,A_s_sh,bs_sh,bpreshuffle=True)
diff4=(ref-out4).abs()
print(f"Both shuffled: max_err={diff4.max().item():.1f}, mean_err={diff4.mean().item():.3f}")

# Fallback preshuffle for tests
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
