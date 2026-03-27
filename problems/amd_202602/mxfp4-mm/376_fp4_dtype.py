#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#376: Check PyTorch FP4 dtypes and try gemm_a4w4_asm with correct dtype.
"""
import torch, os, json
from task import input_t, output_t

# Check available FP4 dtypes
print("=== FP4 dtypes ===")
for name in dir(torch):
    if 'float4' in name.lower() or 'fp4' in name.lower() or 'e2m1' in name.lower():
        print(f"  torch.{name}: {getattr(torch, name)}")

# Check quint4x2
print(f"\ntorch.quint4x2: {torch.quint4x2 if hasattr(torch, 'quint4x2') else 'not found'}")

# Check what dynamic_mxfp4_quant returns
from aiter.ops.triton.quant import dynamic_mxfp4_quant
A = torch.randn(32, 512, dtype=torch.bfloat16, device="cuda")
A_q, A_s = dynamic_mxfp4_quant(A)
print(f"\nA_q dtype: {A_q.dtype}, shape: {A_q.shape}")
print(f"A_s dtype: {A_s.dtype}, shape: {A_s.shape}")

# Check if we can view as a different dtype
try:
    A_q_f4 = A_q.view(torch.float8_e4m3fn)
    print(f"view as float8_e4m3fn: {A_q_f4.dtype}")
except: pass

# List ALL torch.ops.aiter functions
print("\n=== ALL torch.ops.aiter ===")
import aiter
all_ops = sorted(dir(torch.ops.aiter))
for op in all_ops:
    if not op.startswith('_'):
        print(f"  {op}")

# Find the preshuffle loadName
print("\n=== Finding preshuffle loadName ===")
import aiter.ops.triton.gemm.basic.gemm_a16wfp4 as wmod
fn = wmod.gemm_a16wfp4_preshuffle
if hasattr(fn, '__wrapped__'):
    print(f"  wrapped: {fn.__wrapped__}")
if hasattr(fn, '__closure__'):
    for i, cell in enumerate(fn.__closure__ or []):
        try:
            val = cell.cell_contents
            if isinstance(val, str):
                print(f"  closure[{i}]: '{val}'")
        except: pass

# Also check gemm_a16wfp4_preshuffle_
fn2 = wmod.gemm_a16wfp4_preshuffle_
if hasattr(fn2, '__closure__'):
    for i, cell in enumerate(fn2.__closure__ or []):
        try:
            val = cell.cell_contents
            if isinstance(val, str):
                print(f"  _closure[{i}]: '{val}'")
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
