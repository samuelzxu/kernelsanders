#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#265: Extract the actual preshuffle Triton kernel source from the runner.
Dump the full source code of _gemm_a16wfp4_preshuffle_kernel and its dependencies.
"""
import os, json, sys, torch, inspect
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

# Extract kernel source
try:
    import aiter.ops.triton.gemm.basic.gemm_a16wfp4 as wrapper_mod
    # Get the wrapper function source
    wrapper_src = inspect.getsource(wrapper_mod.gemm_a16wfp4_preshuffle_)
    print(f"[265] WRAPPER SOURCE ({len(wrapper_src)} chars):", file=sys.stderr)
    print(wrapper_src[:2000], file=sys.stderr)
except Exception as e:
    print(f"[265] wrapper source error: {e}", file=sys.stderr)

try:
    # Get the kernel JIT function source
    kern = wrapper_mod._gemm_a16wfp4_preshuffle_kernel
    kern_fn = kern.fn if hasattr(kern, 'fn') else kern
    kern_src = inspect.getsource(kern_fn.fn if hasattr(kern_fn, 'fn') else kern_fn)
    print(f"[265] KERNEL SOURCE ({len(kern_src)} chars):", file=sys.stderr)
    print(kern_src[:3000], file=sys.stderr)
except Exception as e:
    print(f"[265] kernel source error: {e}", file=sys.stderr)

try:
    # Get _get_config source
    get_cfg_src = inspect.getsource(wrapper_mod._get_config)
    print(f"[265] _GET_CONFIG ({len(get_cfg_src)} chars):", file=sys.stderr)
    print(get_cfg_src[:1500], file=sys.stderr)
except Exception as e:
    print(f"[265] get_config error: {e}", file=sys.stderr)

# Also get the quant op source
try:
    import aiter.ops.triton._triton_kernels.quant.quant as quant_mod
    for name in dir(quant_mod):
        if 'mxfp4' in name.lower() or 'quant_op' in name.lower():
            obj = getattr(quant_mod, name)
            try:
                src = inspect.getsource(obj)
                print(f"[265] {name} ({len(src)} chars):", file=sys.stderr)
                print(src[:1500], file=sys.stderr)
            except:
                print(f"[265] {name}: no source", file=sys.stderr)
except Exception as e:
    print(f"[265] quant source error: {e}", file=sys.stderr)

# Fallback
_ck = None; _cw = None; _cs = None
@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _ck, _cw, _cs
    A = data[0]; B_shuffle = data[3]; B_scale_sh = data[4]
    m, k = A.shape; n = data[1].shape[0]
    dp = B_shuffle.data_ptr()
    if dp != _ck:
        _ck = dp
        _cw = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        _cs = B_scale_sh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
    return gemm_a16wfp4_preshuffle(A, _cw, _cs, prequant=True, dtype=torch.bfloat16)
