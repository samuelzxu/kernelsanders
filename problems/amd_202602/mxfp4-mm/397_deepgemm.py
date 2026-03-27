#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#397: Investigate deepgemm and deepgemm_ck. What do they do?
"""
import torch, inspect
from task import input_t, output_t
import aiter

# deepgemm source
print("=== deepgemm ===")
try:
    src = inspect.getsource(aiter.deepgemm)
    print(src[:2000])
except Exception as e:
    print(f"Error: {e}")

# deepgemm_ck source
print("\n=== deepgemm_ck ===")
try:
    src = inspect.getsource(aiter.deepgemm_ck)
    print(src[:2000])
except Exception as e:
    print(f"Error: {e}")

# Also check: is there a gemm_afp4wfp4_pre_quant_atomic?
print("\n=== afp4wfp4 variants ===")
try:
    import aiter.ops.triton.gemm.basic.gemm_afp4wfp4 as afmod
    for name in sorted(dir(afmod)):
        if not name.startswith('_') and 'gemm' in name.lower():
            print(f"  {name}")
except: print("  not found")

# Check gemm_a16wfp4 — any other functions besides preshuffle?
print("\n=== a16wfp4 module contents ===")
try:
    import aiter.ops.triton.gemm.basic.gemm_a16wfp4 as wmod
    for name in sorted(dir(wmod)):
        if not name.startswith('__'):
            print(f"  {name}")
except: pass

# Also: the Gluon kernel path
print("\n=== Gluon kernel ===")
try:
    import aiter.ops.triton.gluon as gmod
    for name in sorted(dir(gmod)):
        if 'gemm' in name.lower() or 'fp4' in name.lower():
            print(f"  {name}")
except: print("  gluon module not importable")

try:
    import aiter.ops.triton.gluon.gemm_afp4wfp4 as gfmod
    for name in sorted(dir(gfmod)):
        if not name.startswith('__'):
            print(f"  gluon: {name}")
except: print("  gluon.gemm_afp4wfp4 not importable")

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
