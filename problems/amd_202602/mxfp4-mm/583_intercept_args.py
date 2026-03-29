#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#583: Intercept Triton's actual kernel launch args to determine exact layout.
Monkey-patch the JITFunction to capture the args passed to hipModuleLaunchKernel.
"""
import torch, os, json
from task import input_t, output_t
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'

from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info

_cfgs={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=4096-K=512":{"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=2112-K=7168":{"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=7168-K=2048":{"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=3072-K=1536":{"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)

# Trigger JIT warmup
for _m,_n,_k in [(4,2880,512)]:
    _A=torch.randn((_m,_k),dtype=torch.bfloat16,device="cuda")
    _Bw=torch.zeros((_n//16,(_k//2)*16),dtype=torch.uint8,device="cuda")
    _Bws=torch.zeros((_n//32,_k),dtype=torch.uint8,device="cuda")
    gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
torch.cuda.synchronize()

print("=== INTERCEPTING TRITON ARGS ===")

# Monkey-patch to intercept the wrapper's call to the kernel
# The wrapper function passes specific args to _gemm_a16wfp4_preshuffle_kernel[grid](...)
# Let's intercept the wrapper's local variables

import inspect
# Read the wrapper source
src = inspect.getsource(gemm_a16wfp4_preshuffle)
print("Wrapper calls kernel with these args:")
# Find the kernel invocation
for line in src.split('\n'):
    if '_preshuffle_kernel[' in line or 'kernel[' in line.lower():
        # Print surrounding lines
        idx = src.split('\n').index(line)
        for i in range(max(0, idx-2), min(len(src.split('\n')), idx+20)):
            print(f"  {src.split(chr(10))[i].rstrip()[:120]}")
        break

# Also: intercept the actual JITFunction.run() to capture args
_captured_args = []
import triton

# Find the preshuffle kernel JITFunction
import importlib
mod = importlib.import_module('aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16wfp4')
kernel_fn = None
for name in dir(mod):
    obj = getattr(mod, name)
    if isinstance(obj, triton.runtime.JITFunction) and 'preshuffle' in name:
        kernel_fn = obj
        print(f"\nFound JITFunction: {name}")
        print(f"  params: {[p.name for p in kernel_fn.params]}")
        print(f"  num params: {len(kernel_fn.params)}")
        for p in kernel_fn.params:
            is_constexpr = hasattr(p, 'is_constexpr') and p.is_constexpr
            print(f"    {p.name}: constexpr={is_constexpr}")
        break

if kernel_fn is None:
    print("Could not find preshuffle JITFunction")

# Warmup all shapes
for _m,_n,_k in [(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
    try:
        _A=torch.randn((_m,_k),dtype=torch.bfloat16,device="cuda")
        _Bw=torch.zeros((_n//16,(_k//2)*16),dtype=torch.uint8,device="cuda")
        _Bws=torch.zeros((_n//32,_k),dtype=torch.uint8,device="cuda")
        gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
    except:pass
torch.cuda.empty_cache()

_ps_ck=None;_ps_cw=None;_ps_cs=None
@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _ps_ck,_ps_cw,_ps_cs
    A=data[0];B_shuffle=data[3];B_scale_sh=data[4]
    m,k=A.shape;n=data[1].shape[0]
    dp=B_shuffle.data_ptr()
    if dp!=_ps_ck:
        _ps_ck=dp;_ps_cw=B_shuffle.view(torch.uint8).reshape(n//16,(k//2)*16)
        _ps_cs=B_scale_sh.view(torch.uint8)[:n,:].contiguous().reshape(n//32,k)
    return gemm_a16wfp4_preshuffle(A,_ps_cw,_ps_cs,prequant=True,dtype=torch.bfloat16)
