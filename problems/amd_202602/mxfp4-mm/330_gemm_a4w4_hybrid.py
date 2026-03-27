#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#330: Use gemm_a4w4 for K=1536 M=256 (separate quant + ASM/blockscale GEMM).
For other shapes use preshuffle. The GEMM-only part of gemm_a4w4 might be
much faster than Triton preshuffle, compensating for quant overhead.

gemm_a4w4 takes: A[M,K//2] FP4, B[N,K//2] FP4, A_scale[M,K//32], B_scale[N,K//32]
We have: data[2]=B_q, data[4]=B_scale_sh (shuffled)
Need: dynamic_mxfp4_quant(A) → A_q, A_scale; then e8m0_shuffle(A_scale)
"""
import os, json, torch
from task import input_t, output_t
from aiter.ops.triton.quant import dynamic_mxfp4_quant

# Import e8m0_shuffle
import aiter
_e8m0_shuffle = aiter.fp4_utils.e8m0_shuffle

# Import gemm_a4w4 and trigger JIT build
print("=== Init gemm_a4w4 ===")
from aiter import gemm_a4w4

# Trigger JIT build with a warmup call
_m, _n, _k = 256, 3072, 1536
_A = torch.randn((_m, _k), dtype=torch.bfloat16, device="cuda")
_A_q, _A_scale = dynamic_mxfp4_quant(_A)
_A_scale_sh = _e8m0_shuffle(_A_scale.contiguous())
_B_q = torch.zeros((_n, _k//2), dtype=torch.uint8, device="cuda")
_B_scale = torch.zeros((_n, _k//32), dtype=torch.uint8, device="cuda")
try:
    _out = gemm_a4w4(_A_q, _B_q, _A_scale_sh, _B_scale, bpreshuffle=True)
    print(f"gemm_a4w4 warmup OK: {_out.shape}")
except Exception as e:
    print(f"gemm_a4w4 warmup error: {e}")
del _A, _A_q, _A_scale, _A_scale_sh, _B_q, _B_scale
torch.cuda.empty_cache()

# Preshuffle for other shapes
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
_cfgs = {"N=2880-K=512": {"M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=4096-K=512": {"M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=2112-K=7168": {"M_LEQ_16": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 8, "num_warps": 4, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=7168-K=2048": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=3072-K=1536": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}}
try: _dev = arch_info.get_arch()
except: _dev = "gfx950"
_cd = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
os.makedirs(_cd, exist_ok=True)
for _sk, _cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json", "w") as f:
        json.dump(_cfg, f)

for _m,_n,_k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048)]:
    try:
        _A=torch.randn((_m,_k),dtype=torch.bfloat16,device="cuda")
        _Bw=torch.zeros((_n//16,(_k//2)*16),dtype=torch.uint8,device="cuda")
        _Bws=torch.zeros((_n//32,_k),dtype=torch.uint8,device="cuda")
        gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
    except: pass
torch.cuda.empty_cache()

_ck=None;_cw=None;_cs=None

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _ck,_cw,_cs
    A=data[0]; m,k=A.shape; n=data[1].shape[0]

    if m==256 and k==1536:
        # gemm_a4w4 path for K=1536 M=256
        B_q = data[2]  # [N, K//2] FP4
        B_scale_sh = data[4]  # [N_pad, K//32] shuffled E8M0

        # Quant A
        A_q, A_scale = dynamic_mxfp4_quant(A)
        A_scale_sh = _e8m0_shuffle(A_scale.contiguous())

        # Call gemm_a4w4 with preshuffle B
        return gemm_a4w4(A_q, B_q, A_scale_sh, B_scale_sh[:n,:].contiguous(), bpreshuffle=True)
    else:
        # Preshuffle for other shapes
        B_shuffle=data[3]; B_scale_sh=data[4]
        dp=B_shuffle.data_ptr()
        if dp!=_ck:
            _ck=dp;_cw=B_shuffle.view(torch.uint8).reshape(n//16,(k//2)*16)
            _cs=B_scale_sh.view(torch.uint8)[:n,:].contiguous().reshape(n//32,k)
        return gemm_a16wfp4_preshuffle(A,_cw,_cs,prequant=True,dtype=torch.bfloat16)
