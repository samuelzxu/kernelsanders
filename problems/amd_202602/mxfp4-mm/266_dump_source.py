#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#266: Dump FULL kernel source to stdout (captured by popcorn).
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

# Dump ALL source to stderr (full, not truncated)
import aiter.ops.triton.gemm.basic.gemm_a16wfp4 as wmod
kern = wmod._gemm_a16wfp4_preshuffle_kernel
kern_fn = kern.fn if hasattr(kern, 'fn') else kern
actual_fn = kern_fn.fn if hasattr(kern_fn, 'fn') else kern_fn

# Get full kernel source
try:
    src = inspect.getsource(actual_fn)
    print("===KERNEL_SOURCE_START===", file=sys.stderr)
    print(src, file=sys.stderr)
    print("===KERNEL_SOURCE_END===", file=sys.stderr)
except Exception as e:
    print(f"kernel source error: {e}", file=sys.stderr)

# Get wrapper source
try:
    src = inspect.getsource(wmod.gemm_a16wfp4_preshuffle_)
    print("===WRAPPER_SOURCE_START===", file=sys.stderr)
    print(src, file=sys.stderr)
    print("===WRAPPER_SOURCE_END===", file=sys.stderr)
except Exception as e:
    print(f"wrapper source error: {e}", file=sys.stderr)

# Get _get_config
try:
    src = inspect.getsource(wmod._get_config)
    print("===GETCONFIG_SOURCE_START===", file=sys.stderr)
    print(src, file=sys.stderr)
    print("===GETCONFIG_SOURCE_END===", file=sys.stderr)
except Exception as e:
    print(f"getconfig source error: {e}", file=sys.stderr)

# Get reduce kernel
try:
    src = inspect.getsource(wmod._gemm_afp4wfp4_reduce_kernel.fn.fn if hasattr(wmod._gemm_afp4wfp4_reduce_kernel, 'fn') else wmod._gemm_afp4wfp4_reduce_kernel)
    print("===REDUCE_SOURCE_START===", file=sys.stderr)
    print(src, file=sys.stderr)
    print("===REDUCE_SOURCE_END===", file=sys.stderr)
except Exception as e:
    print(f"reduce source error: {e}", file=sys.stderr)

# Get the _mxfp4_quant_op if accessible through the kernel module
try:
    import aiter.ops.triton._triton_kernels.quant.quant as qmod
    for name in ['_mxfp4_quant_op', '_dynamic_mxfp4_quant_kernel', 'dynamic_mxfp4_quant']:
        if hasattr(qmod, name):
            obj = getattr(qmod, name)
            fn = obj.fn if hasattr(obj, 'fn') else obj
            fn = fn.fn if hasattr(fn, 'fn') else fn
            try:
                src = inspect.getsource(fn)
                print(f"==={name}_SOURCE_START===", file=sys.stderr)
                print(src, file=sys.stderr)
                print(f"==={name}_SOURCE_END===", file=sys.stderr)
            except:
                print(f"{name}: type={type(fn)}, no source", file=sys.stderr)
except Exception as e:
    print(f"quant module error: {e}", file=sys.stderr)

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
