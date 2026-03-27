#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#340: Debug why K=1536 M=256 fails in benchmark but passes in test.
Run the hiprtc quant + hipBLASLt AND preshuffle for the same data, compare outputs.
"""
import torch, os, json
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
from aiter.ops.triton.quant import dynamic_mxfp4_quant

def e8m0_unshuffle(scale_sh, N, K_groups):
    sm, sn = scale_sh.shape
    t = scale_sh.view(sm // 32, sn // 8, 4, 16, 2, 2)
    t = t.permute(0, 5, 3, 1, 4, 2).contiguous()
    return t.view(sm, sn)[:N, :K_groups].contiguous()

# Import reference implementation
from reference import generate_input

# Generate data for K=1536 M=256 seed=7856
data = generate_input(m=256, n=3072, k=1536, seed=7856)
A, B, B_q, B_shuffle, B_scale_sh = data

print(f"A: {A.shape}, B_q: {B_q.shape}, B_scale_sh: {B_scale_sh.shape}")

# Run preshuffle (reference)
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
_cfgs = {"N=3072-K=1536": {"M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}}
try: _dev = arch_info.get_arch()
except: _dev = "gfx950"
_cd = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
os.makedirs(_cd, exist_ok=True)
for _sk, _cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json", "w") as f:
        json.dump(_cfg, f)

m, k = A.shape; n = B.shape[0]
_cw = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
_cs = B_scale_sh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
ref_out = gemm_a16wfp4_preshuffle(A, _cw, _cs, prequant=True, dtype=torch.bfloat16)

# Run hiprtc quant + hipBLASLt
# First: dynamic_mxfp4_quant + hipBLASLt (known to work)
A_q_ref, A_scale_ref = dynamic_mxfp4_quant(A)
print(f"A_q_ref: {A_q_ref.shape}, A_scale_ref: {A_scale_ref.shape} strides={A_scale_ref.stride()}")
B_scale_raw = e8m0_unshuffle(B_scale_sh, n, k // 32)

# Check: does A_scale need contiguous for hipBLASLt?
print(f"A_scale_ref contiguous: {A_scale_ref.is_contiguous()}")
A_scale_c = A_scale_ref.contiguous()

# Now check: our hiprtc quant output
# TODO: compare hiprtc quant output with dynamic_mxfp4_quant output

# For now just check: does dynamic_mxfp4_quant + unshuffle + hipBLASLt give correct output?
# This was passing before. If it fails here, the issue is shape-specific.

# Fallback to preshuffle for correctness
_ck=None;_cw2=None;_cs2=None
@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _ck,_cw2,_cs2
    A=data[0];B_shuffle=data[3];B_scale_sh=data[4]
    m,k=A.shape;n=data[1].shape[0]
    dp=B_shuffle.data_ptr()
    if dp!=_ck:
        _ck=dp;_cw2=B_shuffle.view(torch.uint8).reshape(n//16,(k//2)*16)
        _cs2=B_scale_sh.view(torch.uint8)[:n,:].contiguous().reshape(n//32,k)
    return gemm_a16wfp4_preshuffle(A,_cw2,_cs2,prequant=True,dtype=torch.bfloat16)
