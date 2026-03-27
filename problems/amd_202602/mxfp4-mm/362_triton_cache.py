#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#362: Probe Triton cache to find compiled HSACO kernels.
If we can launch them directly via hipModule, we skip Triton dispatch (~2-3µs).
"""
import torch, os, glob, json
from task import input_t, output_t
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info

_cfgs={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)

# Warm up to compile
_A=torch.randn((4,512),dtype=torch.bfloat16,device="cuda")
_Bw=torch.zeros((2880//16,(512//2)*16),dtype=torch.uint8,device="cuda")
_Bws=torch.zeros((2880//32,512),dtype=torch.uint8,device="cuda")
gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)

# Find Triton cache
print("=== Triton cache ===")
cache_dirs = glob.glob(os.path.expanduser("~/.triton/cache/*"))
for d in sorted(cache_dirs)[:5]:
    print(f"  {d}")
    files = glob.glob(f"{d}/**/*", recursive=True)
    for f in sorted(files)[:10]:
        sz = os.path.getsize(f) if os.path.isfile(f) else 0
        print(f"    {os.path.basename(f)} ({sz} bytes)")

# Check for .hsaco or compiled kernel files
print("\n=== HSACO files ===")
for f in glob.glob(os.path.expanduser("~/.triton/cache/**/*.hsaco"), recursive=True):
    print(f"  {f} ({os.path.getsize(f)} bytes)")

# Check for .json metadata
print("\n=== Kernel metadata ===")
for f in sorted(glob.glob(os.path.expanduser("~/.triton/cache/**/*.json"), recursive=True))[:5]:
    try:
        with open(f) as fh:
            data = json.load(fh)
        print(f"  {os.path.basename(f)}: {json.dumps(data)[:200]}")
    except: pass

# Check Triton compiled kernel objects
print("\n=== Triton kernel info ===")
import aiter.ops.triton.gemm.basic.gemm_a16wfp4 as wmod
kern = wmod._gemm_a16wfp4_preshuffle_kernel
print(f"Kernel type: {type(kern)}")
if hasattr(kern, 'cache'):
    print(f"Cache keys: {list(kern.cache.keys())[:3]}")
    for k, v in list(kern.cache.items())[:1]:
        print(f"  Key: {k}")
        print(f"  Value type: {type(v)}")
        for attr in ['asm', 'metadata', 'name', 'num_warps', 'shared']:
            if hasattr(v, attr):
                val = getattr(v, attr)
                if isinstance(val, dict):
                    print(f"  .{attr}: {list(val.keys())[:5]}")
                elif isinstance(val, str) and len(val) > 100:
                    print(f"  .{attr}: ({len(val)} chars)")
                else:
                    print(f"  .{attr}: {val}")

# Standard kernel
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
