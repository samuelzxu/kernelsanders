#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#399: Try gemm_afp4wfp4_preshuffle — takes pre-quantized FP4 A + preshuffle B.
This is a DIFFERENT Triton kernel from gemm_a16wfp4_preshuffle.
If it's faster (no inline quant), total = quant + gemm might beat fused.
"""
import torch, os, json, inspect, time
from task import input_t, output_t
from aiter.ops.triton.quant import dynamic_mxfp4_quant

# Check gemm_afp4wfp4_preshuffle
print("=== gemm_afp4wfp4_preshuffle ===")
import aiter.ops.triton.gemm.basic.gemm_afp4wfp4 as afmod
try:
    fn = afmod.gemm_afp4wfp4_preshuffle
    print(f"Type: {type(fn)}")
    # Try to get signature
    try:
        src = inspect.getsource(fn)
        print(src[:1500])
    except:
        # Check closure for loadName
        if hasattr(fn, '__closure__'):
            for i, cell in enumerate(fn.__closure__ or []):
                try:
                    val = cell.cell_contents
                    if isinstance(val, str):
                        print(f"  closure[{i}]: '{val}'")
                except: pass
except Exception as e:
    print(f"Error: {e}")

# Check gemm_afp4wfp4_ (underlying)
print("\n=== gemm_afp4wfp4_ ===")
try:
    fn2 = afmod.gemm_afp4wfp4_
    if hasattr(fn2, '__closure__'):
        for i, cell in enumerate(fn2.__closure__ or []):
            try:
                val = cell.cell_contents
                if isinstance(val, str):
                    print(f"  closure[{i}]: '{val}'")
            except: pass
except: pass

# Try calling gemm_afp4wfp4_preshuffle
print("\n=== Test call ===")
M,N,K=256,3072,1536
A=torch.randn(M,K,dtype=torch.bfloat16,device="cuda")
A_q,A_s=dynamic_mxfp4_quant(A)
A_s=A_s.contiguous()
import aiter
A_s_sh=aiter.fp4_utils.e8m0_shuffle(A_s)

B_q=torch.zeros(N,K//2,dtype=torch.uint8,device="cuda")
# Preshuffle B format
B_w=B_q.view(torch.uint8).reshape(N//16,(K//2)*16)
B_scale=torch.zeros(N//32,K,dtype=torch.uint8,device="cuda")

try:
    result=afmod.gemm_afp4wfp4_preshuffle(A_q,B_w,A_s_sh,B_scale,dtype=torch.bfloat16)
    print(f"SUCCESS: {result.shape}")
except Exception as e:
    print(f"Error: {str(e)[:200]}")

# Try with different arg combos
for desc,args,kw in [
    ("Aq,Bw,As,Bs", (A_q,B_w,A_s,B_scale), {"dtype":torch.bfloat16}),
    ("Aq,Bw,As_sh,Bs", (A_q,B_w,A_s_sh,B_scale), {"dtype":torch.bfloat16}),
    ("Aq,Bq,As,Bs", (A_q,B_q,A_s,B_scale), {"dtype":torch.bfloat16}),
]:
    try:
        r=afmod.gemm_afp4wfp4_preshuffle(*args,**kw)
        print(f"  {desc}: OK {r.shape}")
    except Exception as e:
        print(f"  {desc}: {str(e)[:100]}")

# Standard kernel
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
_cfgs={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
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
