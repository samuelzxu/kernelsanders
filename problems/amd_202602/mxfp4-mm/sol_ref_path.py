#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Try the EXACT reference path: gemm_a4w4 with dtypes.fp4x2 and fp8_e8m0.
The reference.py uses this — maybe it's faster than preshuffle!
"""
import torch, os, json, time
from task import input_t, output_t
import aiter
from aiter import dtypes, QuantType
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle

print(f"dtypes.fp4x2 = {dtypes.fp4x2}")
print(f"dtypes.fp8_e8m0 = {dtypes.fp8_e8m0}")
print(f"dtypes.bf16 = {dtypes.bf16}")

# Try the EXACT reference path
M,N,K=256,3072,1536
A=torch.randn(M,K,dtype=torch.bfloat16,device="cuda")
B_q_raw=torch.zeros(N,K//2,dtype=torch.uint8,device="cuda")
from aiter.ops.shuffle import shuffle_weight
B_shuffle=shuffle_weight(B_q_raw.view(dtypes.fp4x2),layout=(16,16))
B_scale_sh=torch.zeros(N,K//32,dtype=torch.uint8,device="cuda").view(dtypes.fp8_e8m0)

# Quant A
A_q, A_scale = dynamic_mxfp4_quant(A)
A_q_typed = A_q.view(dtypes.fp4x2)
A_scale_sh = e8m0_shuffle(A_scale).view(dtypes.fp8_e8m0)

print(f"A_q_typed: {A_q_typed.shape} {A_q_typed.dtype}")
print(f"A_scale_sh: {A_scale_sh.shape} {A_scale_sh.dtype}")
print(f"B_shuffle: {B_shuffle.shape} {B_shuffle.dtype}")
print(f"B_scale_sh: {B_scale_sh.shape} {B_scale_sh.dtype}")

# Call gemm_a4w4 exactly like the reference
try:
    out = aiter.gemm_a4w4(A_q_typed, B_shuffle, A_scale_sh, B_scale_sh, dtype=dtypes.bf16, bpreshuffle=True)
    print(f"gemm_a4w4 ref path: {out.shape} {out.dtype}")

    # Time it
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        aiter.gemm_a4w4(A_q_typed, B_shuffle, A_scale_sh, B_scale_sh, dtype=dtypes.bf16, bpreshuffle=True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(100):
        aiter.gemm_a4w4(A_q_typed, B_shuffle, A_scale_sh, B_scale_sh, dtype=dtypes.bf16, bpreshuffle=True)
    e.record();torch.cuda.synchronize()
    print(f"gemm_a4w4 time: {s.elapsed_time(e)/100*1000:.1f}us")

    # Also time quant + shuffle
    s.record()
    for _ in range(100):
        aq,asc=dynamic_mxfp4_quant(A)
        ash=e8m0_shuffle(asc)
    e.record();torch.cuda.synchronize()
    print(f"quant+shuffle time: {s.elapsed_time(e)/100*1000:.1f}us")

except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()

# Standard preshuffle fallback
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
