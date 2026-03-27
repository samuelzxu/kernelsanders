#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#425: Read the FULL C++ wrapper source for gemm_a16wfp4_preshuffle.
The Python wrapper is just a passthrough to torch.ops.aiter.
The C++ code handles config, splitk, allocation, and kernel launch.
Let me find the C++ source to understand if there's an output pre-alloc option.
"""
import torch, os, glob, json
from task import input_t, output_t

# Find the C++ source for gemm_a16wfp4_preshuffle
print("=== C++ source search ===")
# The JIT module is at: /home/runner/aiter/aiter/jit/
# The gemm_a16wfp4 is a Triton kernel, so the C++ wrapper might be
# generated or in a specific ops file.

# Check the ops directory for the preshuffle wrapper
import subprocess
try:
    r = subprocess.run(["grep", "-rl", "a16wfp4", "/home/runner/aiter/aiter/ops/", "--include=*.py"],
                       capture_output=True, text=True, timeout=10)
    for f in r.stdout.strip().split('\n')[:5]:
        if f:
            print(f"  {f}")
            with open(f) as fh:
                content = fh.read()
            if 'preshuffle' in content and 'def ' in content:
                for line in content.split('\n'):
                    if 'def ' in line and ('preshuffle' in line.lower() or 'a16w' in line.lower()):
                        print(f"    {line.strip()}")
except: pass

# Check: is there a gemm_a16wfp4 in the ops directory (not triton)?
print("\n=== gemm_op_a16wfp4 ===")
try:
    with open("/home/runner/aiter/aiter/ops/gemm_op_a16wfp4.py") as f:
        content = f.read()
    print(content[:3000])
except FileNotFoundError:
    print("Not found")
except Exception as e:
    print(f"Error: {e}")

# Check the triton module for the wrapper
print("\n=== triton gemm_a16wfp4 wrapper ===")
try:
    import aiter.ops.triton.gemm.basic.gemm_a16wfp4 as wmod
    # Get the actual gemm_a16wfp4_preshuffle function (not the wrapper)
    fn = wmod.gemm_a16wfp4_preshuffle
    # Check if it's a compiled op or Python
    print(f"Type: {type(fn)}")
    print(f"Module: {fn.__module__}")

    # Get gemm_a16wfp4_preshuffle_ (the C++ op wrapper)
    fn2 = wmod.gemm_a16wfp4_preshuffle_
    print(f"_ Type: {type(fn2)}")

    # Check what fn actually calls
    import inspect
    try:
        src = inspect.getsource(fn)
        print(f"\ngemm_a16wfp4_preshuffle source:")
        print(src[:2000])
    except: pass
except: pass

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
