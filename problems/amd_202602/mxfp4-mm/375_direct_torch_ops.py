#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#375: Probe torch.ops.aiter for the direct C++ function signature.
Can we pass pre-allocated output? What parameters does it take?
"""
import torch, os, json
from task import input_t, output_t

# Find the actual loadName for gemm_a16wfp4_preshuffle
print("=== torch.ops.aiter functions ===")
import aiter
ops = sorted([x for x in dir(torch.ops.aiter) if 'preshuffle' in x.lower() or 'a16w' in x.lower()])
print(f"Preshuffle ops: {ops}")

# Try calling with extra output argument
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info

_cfgs={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)

# Warm up
_A=torch.randn((4,512),dtype=torch.bfloat16,device="cuda")
_Bw=torch.zeros((2880//16,(512//2)*16),dtype=torch.uint8,device="cuda")
_Bws=torch.zeros((2880//32,512),dtype=torch.uint8,device="cuda")
result = gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
print(f"Normal call result: {result.shape}")

# Try torch.ops.aiter directly
for name in ops:
    fn = getattr(torch.ops.aiter, name)
    print(f"\n=== {name} ===")
    print(f"Type: {type(fn)}")
    # Try different signatures
    for desc, args in [
        ("A,Bw,Bws,True,bf16", (_A, _Bw, _Bws, True, torch.bfloat16)),
        ("A,Bw,Bws,True,bf16,out", (_A, _Bw, _Bws, True, torch.bfloat16, result)),
    ]:
        try:
            r = fn(*args)
            print(f"  {desc}: OK, type={type(r)}, shape={r.shape if hasattr(r,'shape') else '?'}")
        except Exception as e:
            print(f"  {desc}: {str(e)[:100]}")

# Check what the schema says
for name in ops:
    try:
        schemas = torch.ops.aiter._overloadpacket_for(name)
        print(f"\n{name} schemas: {schemas}")
    except: pass

# Check if there's a schema via _schemas
try:
    for name in ops:
        fn = getattr(torch.ops.aiter, name)
        if hasattr(fn, '_schemas'):
            print(f"{name} schemas: {fn._schemas}")
        if hasattr(fn, 'default'):
            d = fn.default
            if hasattr(d, '_schema'):
                print(f"{name}.default schema: {d._schema}")
except: pass

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
