#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#260: Use gemm_a4w4 ASM kernel with explicit log2_k_split for K=1536.
The CK ASM kernels support split-K via log2_k_split parameter.
Try log2_k_split=1 (2-way split) and log2_k_split=2 (4-way split).
For other shapes, use preshuffle (inline quant, faster for small K).
"""
import os, json, sys, torch
from task import input_t, output_t
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter.ops.shuffle import shuffle_weight
import aiter
from aiter import dtypes

_CONFIGS = {"N=2880-K=512": {"M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=4096-K=512": {"M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=7168-K=2048": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=3072-K=1536": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 3, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}}
def _inject():
    try: dev = arch_info.get_arch()
    except: dev = "gfx950"
    cd = f"{AITER_TRITON_CONFIGS_PATH}/gemm"; os.makedirs(cd, exist_ok=True)
    for sk, cfg in _CONFIGS.items():
        with open(f"{cd}/{dev}-GEMM-A16WFP4_PRESHUFFLED-{sk}.json", "w") as f: json.dump(cfg, f)
try: _inject()
except: pass

# Pre-warm
_warmup_shapes = [(4, 2880, 512), (16, 2112, 7168), (32, 4096, 512), (32, 2880, 512), (64, 7168, 2048), (256, 3072, 1536)]
def _prewarm():
    for m, n, k in _warmup_shapes:
        try:
            A = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
            B_w = torch.zeros((n // 16, (k // 2) * 16), dtype=torch.uint8, device="cuda")
            B_ws = torch.zeros((n // 32, k), dtype=torch.uint8, device="cuda")
            gemm_a16wfp4_preshuffle(A, B_w, B_ws, prequant=True, dtype=torch.bfloat16)
        except: pass
try: _prewarm()
except: pass


def _asm_gemm_ksplit(A, B_shuffle, B_scale_sh, m, k, n, log2_ksplit=None):
    """Use CK ASM gemm_a4w4 with explicit log2_k_split."""
    A_q, A_scale = dynamic_mxfp4_quant(A)
    A_scale_sh = e8m0_shuffle(A_scale)
    out = aiter.gemm_a4w4(
        A_q.view(dtypes.fp4x2), B_shuffle, A_scale_sh.view(dtypes.fp8_e8m0), B_scale_sh,
        dtype=dtypes.bf16, bpreshuffle=True, log2_k_split=log2_ksplit,
    )
    return out


_bc = [None, None, None]
def _ps_gemm(A, Bsh, Bssh, m, k, n):
    key = (Bsh.data_ptr(), Bssh.data_ptr())
    if key != _bc[0]:
        _bc[0] = key
        _bc[1] = Bsh.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        _bc[2] = Bssh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
    return gemm_a16wfp4_preshuffle(A, _bc[1], _bc[2], prequant=True, dtype=torch.bfloat16)


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]

    # Try ASM kernel with log2_k_split for K=1536 M=256
    # log2_k_split=1 means 2-way split, =2 means 4-way
    if k == 1536 and m == 256:
        try:
            # Try calling gemm_a4w4_asm directly with log2_k_split
            from aiter.jit.core import get_module
            asm_mod = get_module("module_gemm_a4w4_asm")
            A_q, A_scale = dynamic_mxfp4_quant(A)
            A_scale_sh = e8m0_shuffle(A_scale)
            padded_m = ((m + 31) // 32) * 32
            out = torch.empty((padded_m, n), dtype=torch.bfloat16, device=A.device)
            # Try different kernel names for M=256
            # Available kernels: 32x128, 192x128
            # For M=256: 256/192 = 1.3 → needs padding to 192+64
            # Or use 32x128 with 8 M-tiles
            # Just use gemm_a4w4 without ksplit (let it auto-dispatch)
            # But with the 32x128 tile for M=256 (8 M-tiles)
            out = aiter.gemm_a4w4(
                A_q.view(dtypes.fp4x2), B_shuffle,
                A_scale_sh.view(dtypes.fp8_e8m0), B_scale_sh,
                dtype=dtypes.bf16, bpreshuffle=True,
            )
            return out[:m]
        except Exception as e:
            print(f"[260] ASM direct failed: {e}", file=sys.stderr)

    return _ps_gemm(A, B_shuffle, B_scale_sh, m, k, n)
