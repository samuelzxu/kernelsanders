#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#281: Feed SYNTHETIC data to the full HIP GEMM kernel (not just a probe).
Create A=all_ones bf16, B_q=all 0x22 (FP4=1.0), B_scale=all 127 (scale=1.0).
Expected output: each element = K (number of FP4 elements) since A=1*B=1*scale=1.
For M=32, N=32, K=64 (Kh=32): expected = 64.
"""
import os, shutil, sys, json, torch
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle

_cfgs = {"N=2880-K=512": {"M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=4096-K=512": {"M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=7168-K=2048": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=3072-K=1536": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 3, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}}
try: _dev = arch_info.get_arch()
except: _dev = "gfx950"
_cd = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
os.makedirs(_cd, exist_ok=True)
for _sk, _cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json", "w") as f:
        json.dump(_cfg, f)

# Import the SAME kernel as #272
_MODULE = 'mfma281'
# Copy the kernel source from 272 but with the byte-level zero init
_HIP_SRC = open('/home/ziggy/devel/kernelsanders/problems/amd_202602/mxfp4-mm/272_hip_sw_quant.py').read()
# Extract just the HIP source between r""" markers
import re
hip_match = re.search(r'_HIP_SRC = r"""(.+?)"""', _HIP_SRC, re.DOTALL)
if hip_match:
    _HIP_CODE = hip_match.group(1)
else:
    _HIP_CODE = ""
    print("[281] Could not extract HIP source!", file=sys.stderr)

# Use a simple test: run the kernel on synthetic data
_bc = [None, None, None]
_first = [True]

for _m, _n, _k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
    try:
        _A = torch.randn((_m, _k), dtype=torch.bfloat16, device="cuda")
        _Bw = torch.zeros((_n//16, (_k//2)*16), dtype=torch.uint8, device="cuda")
        _Bws = torch.zeros((_n//32, _k), dtype=torch.uint8, device="cuda")
        gemm_a16wfp4_preshuffle(_A, _Bw, _Bws, prequant=True, dtype=torch.bfloat16)
    except: pass
torch.cuda.empty_cache()

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]

    if _first[0]:
        _first[0] = False
        # Create synthetic test: A=1.0 bf16, B_q=0x22 (1.0), B_scale=127 (1.0)
        # For a 32x32 tile with K=64 (Kh=32):
        test_m, test_n, test_k = 32, 32, 64
        test_Kh = test_k // 2
        test_Ks = test_k // 32
        syn_A = torch.ones((test_m, test_k), dtype=torch.bfloat16, device='cuda')
        syn_Bq = torch.full((test_n, test_Kh), 0x22, dtype=torch.uint8, device='cuda')
        syn_Bs = torch.full((test_n, test_Ks), 127, dtype=torch.uint8, device='cuda')

        # Import the kernel module
        try:
            from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import _gemm_a16wfp4_preshuffle_kernel
            # Can't easily call raw kernel, use the wrapper instead
            # But we need B in preshuffle format and B_scale in preshuffle format
            # For synthetic uniform data, B_shuffle = B_q (shuffle of all-same is same)
            from aiter.ops.shuffle import shuffle_weight
            from aiter import dtypes
            from aiter.utility.fp4_utils import e8m0_shuffle
            from aiter.ops.triton.quant import dynamic_mxfp4_quant

            # Quantize synthetic A
            syn_Aq, syn_As = dynamic_mxfp4_quant(syn_A)
            syn_Aq_u8 = syn_Aq.view(torch.uint8)
            print(f"[281] syn_Aq[0,:4] = {syn_Aq_u8[0,:4].tolist()}", file=sys.stderr)
            print(f"[281] syn_As[0,:2] = {syn_As.view(torch.uint8)[0,:2].tolist()}", file=sys.stderr)

            # The quant of A=1.0 bf16:
            # max_abs = 1.0. amax_int = 0x3F800000. (0x3F800000 + 0x200000) & 0xFF800000 = 0x40000000.
            # biased_exp = 0x80 = 128. e8m0 = 128-2 = 126. scale = 2^(126-127) = 0.5
            # quant_scale = 2^(-(-1)) = 2.0
            # qx = 1.0 * 2.0 = 2.0. FP4(2.0) = code 4 = 0100.
            # Packed: lo=4, hi=4 → byte = 0x44.
            print(f"[281] Expected A byte: 0x44 = {0x44}", file=sys.stderr)

            # B_q all 0x22: lo=2(1.0), hi=2(1.0). Scale=127(1.0).
            # Each MFMA K-block: 64 values, A all 2.0(scaled by 0.5→1.0), B all 1.0(scaled by 1.0)
            # dot = 64 * 1.0 * 1.0 = 64.0
            # For K=64: 1 MFMA iteration. Expected output = 64.0
            # Actually: A_fp4 = 2.0, A_scale = 0.5. B_fp4 = 1.0, B_scale = 1.0.
            # MFMA computes: sum(A_fp4[k] * B_fp4[k]) * A_scale * B_scale
            # = sum(2.0 * 1.0 for 64 values) * 0.5 * 1.0 = 128 * 0.5 = 64.0
            print(f"[281] Expected output: 64.0", file=sys.stderr)

            # Run preshuffle on synthetic data
            syn_Bq_fp4x2 = syn_Bq.view(dtypes.fp4x2)
            syn_Bsh = shuffle_weight(syn_Bq_fp4x2, layout=(16,16))
            syn_Bsh_u8 = syn_Bsh.view(torch.uint8)
            # B_scale: need shuffled format
            syn_Bs_e8m0 = syn_Bs.view(dtypes.fp8_e8m0)
            # For uniform scales, e8m0_shuffle should give same values
            # Create proper scale tensor
            syn_Bs_padded = torch.full((((test_n+31)//32)*32, test_Ks), 127, dtype=torch.uint8, device='cuda')
            syn_Bs_sh = e8m0_shuffle(syn_Bs_padded.view(dtypes.fp8_e8m0))

            syn_Bw = syn_Bsh_u8.reshape(test_n // 16, (test_Kh) * 16)
            syn_Bws = syn_Bs_sh.view(torch.uint8)[:test_n, :].contiguous().reshape(test_n // 32, test_k)

            ps_result = gemm_a16wfp4_preshuffle(syn_A, syn_Bw, syn_Bws, prequant=True, dtype=torch.bfloat16)
            print(f"[281] Preshuffle synthetic [0,0] = {ps_result[0,0].item()}", file=sys.stderr)
            print(f"[281] Preshuffle synthetic [0,:4] = {ps_result[0,:4].tolist()}", file=sys.stderr)
        except Exception as e:
            print(f"[281] synthetic test error: {e}", file=sys.stderr)
            import traceback; traceback.print_exc(file=sys.stderr)

    dp = B_shuffle.data_ptr()
    if dp != _bc[0]:
        _bc[0] = dp
        _bc[1] = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        _bc[2] = B_scale_sh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
    return gemm_a16wfp4_preshuffle(A, _bc[1], _bc[2], prequant=True, dtype=torch.bfloat16)
