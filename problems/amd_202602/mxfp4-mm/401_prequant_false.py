#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#401: Test prequant=False for gemm_a16wfp4_preshuffle.
Does it skip the inline quant? What input format does it expect?
Also test: what configs does the kernel accept that we haven't tried?
"""
import torch, os, json
from task import input_t, output_t
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
import aiter

# Test prequant=False with pre-quantized A
M,N,K=32,2880,512
A=torch.randn(M,K,dtype=torch.bfloat16,device="cuda")
A_q,A_s=dynamic_mxfp4_quant(A)
A_s_sh=aiter.fp4_utils.e8m0_shuffle(A_s.contiguous())

B_q=torch.zeros(N,K//2,dtype=torch.uint8,device="cuda")
B_w=B_q.view(torch.uint8).reshape(N//16,(K//2)*16)
B_scale=torch.zeros(N//32,K,dtype=torch.uint8,device="cuda")

_cfgs={"N=2880-K=512":{"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)

# First: warmup with prequant=True (normal)
ref = gemm_a16wfp4_preshuffle(A, B_w, B_scale, prequant=True, dtype=torch.bfloat16)
print(f"prequant=True: shape={ref.shape}, ref[0,0]={ref[0,0].item():.2f}")

# Now test prequant=False with bf16 A
try:
    out1 = gemm_a16wfp4_preshuffle(A, B_w, B_scale, prequant=False, dtype=torch.bfloat16)
    print(f"prequant=False bf16 A: shape={out1.shape}, out[0,0]={out1[0,0].item():.2f}")
    diff1 = (ref-out1).abs().max().item()
    print(f"  diff from ref: {diff1:.1f}")
except Exception as e:
    print(f"prequant=False bf16 A: {str(e)[:150]}")

# Test prequant=False with FP4 A (A_q)
try:
    out2 = gemm_a16wfp4_preshuffle(A_q, B_w, B_scale, prequant=False, dtype=torch.bfloat16)
    print(f"prequant=False FP4 A: shape={out2.shape}")
except Exception as e:
    print(f"prequant=False FP4 A: {str(e)[:150]}")

# Check: does the preshuffle kernel's config have a 'kpack' parameter?
print("\n=== Check for kpack ===")
import aiter.ops.triton.gemm.basic.gemm_a16wfp4 as wmod
kern = wmod._gemm_a16wfp4_preshuffle_kernel
if hasattr(kern, 'fn'):
    inner = kern.fn if not hasattr(kern.fn, 'fn') else kern.fn.fn
    import inspect
    try:
        src = inspect.getsource(inner)
        # Search for parameters
        for line in src.split('\n')[:20]:
            if 'def ' in line or 'constexpr' in line.lower():
                print(f"  {line.strip()}")
    except: print("  can't get source")

# Standard kernel
_cfgs2={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
for _sk,_cfg in _cfgs2.items():
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
