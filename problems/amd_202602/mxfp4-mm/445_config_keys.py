#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#445: Check EXACTLY what config keys the preshuffle kernel reads.
Maybe there are keys like 'kpack', 'ATOMIC_ADD', 'serialize_output' etc.
"""
import torch, os, json, inspect
from task import input_t, output_t

# Read the FULL wrapper that processes configs
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
import aiter.ops.triton.gemm.basic.gemm_a16wfp4 as wmod

# Check get_gemm_config and what it returns
print("=== get_gemm_config source ===")
try:
    from aiter.ops.triton.utils._triton import get_gemm_config
    src = inspect.getsource(get_gemm_config)
    print(src[:3000])
except Exception as e:
    print(f"Error: {e}")

# Check the full gemm_a16wfp4_preshuffle_ source to see what config keys it uses
print("\n=== Full preshuffle_ wrapper ===")
try:
    fn = wmod.gemm_a16wfp4_preshuffle_
    # Get the actual function, not the closure wrapper
    src = inspect.getsource(fn)
    print(src[:3000])
except:
    # Try the underlying module function
    try:
        # The function might be loaded from a compiled module
        print(f"Type: {type(fn)}")
        if hasattr(fn, '__closure__'):
            for i, cell in enumerate(fn.__closure__ or []):
                try:
                    v = cell.cell_contents
                    if callable(v):
                        src = inspect.getsource(v)
                        print(f"Closure[{i}] source ({len(src)} chars):")
                        print(src[:2000])
                except: pass
    except: pass

# Check: does the config dict get passed to the Triton kernel?
# Or does the wrapper extract specific keys?
print("\n=== gemm_a16wfp4_preshuffle full source ===")
try:
    src = inspect.getsource(wmod.gemm_a16wfp4_preshuffle)
    print(src)
except: pass

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
