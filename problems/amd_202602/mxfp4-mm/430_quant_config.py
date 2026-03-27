#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#430: Check if dynamic_mxfp4_quant has tunable configs.
Also: measure quant time with CUDA events for each shape.
Key question: is the 20µs quant time from Triton dispatch or actual GPU work?
"""
import torch, os, json, inspect, glob
from task import input_t, output_t
from aiter.ops.triton.quant import dynamic_mxfp4_quant

# Check quant function signature and source
print("=== dynamic_mxfp4_quant ===")
try:
    src = inspect.getsource(dynamic_mxfp4_quant)
    print(src[:2000])
except: print("can't get source (compiled op)")

# Check for quant config files
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
print(f"\n=== Quant configs in {AITER_TRITON_CONFIGS_PATH} ===")
for f in sorted(glob.glob(f"{AITER_TRITON_CONFIGS_PATH}/**/*quant*", recursive=True)):
    print(f"  {f}")
for f in sorted(glob.glob(f"{AITER_TRITON_CONFIGS_PATH}/**/*mxfp*", recursive=True)):
    print(f"  {f}")

# Check the _dynamic_mxfp4_quant_kernel source
print("\n=== _dynamic_mxfp4_quant_kernel ===")
try:
    import aiter.ops.triton._triton_kernels.quant.quant as qmod
    if hasattr(qmod, '_dynamic_mxfp4_quant_kernel'):
        fn = qmod._dynamic_mxfp4_quant_kernel
        for _ in range(3):
            if hasattr(fn, 'fn'): fn = fn.fn
        src = inspect.getsource(fn)
        # Print just the signature and config
        lines = src.split('\n')
        for line in lines[:30]:
            print(f"  {line}")
except Exception as e:
    print(f"  Error: {e}")

# Measure quant time per shape with CUDA events
print("\n=== Quant timing ===")
for m,k in [(4,512),(16,7168),(32,512),(64,2048),(256,1536)]:
    A = torch.randn(m,k,dtype=torch.bfloat16,device="cuda")
    # Warmup
    for _ in range(5): dynamic_mxfp4_quant(A)
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(100): dynamic_mxfp4_quant(A)
    e.record();torch.cuda.synchronize()
    print(f"  M={m:3d} K={k:4d}: {s.elapsed_time(e)/100*1000:.1f}us")

# Standard kernel
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
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
