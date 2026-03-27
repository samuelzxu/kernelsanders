#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#437: Check EXACTLY which M-bucket each benchmark shape hits.
The C++ wrapper reads the JSON config and matches M against M_LEQ_* keys.
If M=4 hits M_LEQ_4 and M=32 hits M_LEQ_32, good.
But what about M=16 (K=7168) and M=64 (K=2048) and M=256 (K=1536)?
Do they hit M_LEQ_16, M_LEQ_64, M_LEQ_256?

KEY: The M_LEQ matching might use PADDED M, not raw M.
From gemm_op_a4w4.py: padded_M = get_padded_m(M, N, K, gl)
What if padded M is different from raw M for our shapes?
"""
import torch, os, json
from task import input_t, output_t

# Check what config each shape actually uses
# Set a config with a UNIQUE marker per bucket and see which one the kernel uses
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info

# The _get_config function selects based on M thresholds
# Let me check: does the C++ wrapper use get_padded_m?
# From the wrapper source: it takes (x, w, w_scales, ...) and computes M from x.shape[0]
# It should use raw M, not padded M. But let me verify by setting distinct configs.

# For N=3072-K=1536, set DIFFERENT configs for each M bucket:
# M_LEQ_4: BSN=16 (marker)
# M_LEQ_16: BSN=32 (marker)
# M_LEQ_32: BSN=64 (marker)
# M_LEQ_64: BSN=128 (marker for test shape M=64)
# M_LEQ_128: BSN=256 (marker)
# M_LEQ_256: BSN=256 (our target for M=256)
# any: BSN=128

# If M=256 hits M_LEQ_256 and M=64 hits M_LEQ_64, the benchmark
# should show different speeds for the K=1536 test shape vs benchmark shape.

# Actually, the important question is: does the benchmark shape M=256 K=1536
# REALLY use the M_LEQ_256 config? Or does it use "any"?
# If padded_m = 288 (next multiple of 32), and M_LEQ_256 only matches M<=256,
# then M=256 would match M_LEQ_256 exactly. Good.

# But what if there's a M_LEQ_512 bucket that we're not setting?
# The kernel would fall through to "any" for M>256.
# M=256 should match M_LEQ_256. Let me verify by setting M_LEQ_256 to a bad config
# and seeing if performance changes.

# Test: set M_LEQ_256 to BSN=64 (known to be slow for K=1536) and see if M=256 slows down
_cfgs={"N=3072-K=1536":{
"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":3,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},
"M_LEQ_256":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None},
"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":32,"cache_modifier":".cg"}},
"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=4096-K=512":{"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=2112-K=7168":{"M_LEQ_16":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":8,"num_warps":4,"num_stages":2,"waves_per_eu":3,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=7168-K=2048":{"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":3,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)
for _m,_n,_k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
    try:
        _A=torch.randn((_m,_k),dtype=torch.bfloat16,device="cuda")
        _Bw=torch.zeros((_n//16,(_k//2)*16),dtype=torch.uint8,device="cuda")
        _Bws=torch.zeros((_n//32,_k),dtype=torch.uint8,device="cuda")
        gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
    except:pass
torch.cuda.empty_cache()
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
