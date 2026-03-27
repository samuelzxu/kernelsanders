#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4-MM #255: hipBLASLt FP4 GEMM with VEC32_UE8M0 block scaling.
Probe whether hipBLASLt supports MXFP4 on gfx950 runner.
Uses load_inline to call hipBLASLt C API directly.
"""
import os, shutil, sys, json, torch
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
os.environ['HSA_XNACK'] = '0'

from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.quant import dynamic_mxfp4_quant

_CONFIGS = {"N=2880-K=512": {"M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=4096-K=512": {"M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=7168-K=2048": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=3072-K=1536": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 3, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}}
def _inject():
    try: dev = arch_info.get_arch()
    except: dev = "gfx950"
    cd = f"{AITER_TRITON_CONFIGS_PATH}/gemm"; os.makedirs(cd, exist_ok=True)
    for sk, cfg in _CONFIGS.items():
        with open(f"{cd}/{dev}-GEMM-A16WFP4_PRESHUFFLED-{sk}.json", "w") as f: json.dump(cfg, f)
try: _inject()
except: pass

# hipBLASLt probe
_MODULE = 'hipblaslt_probe_255'
_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <torch/extension.h>

std::string probe_hipblaslt() {
    std::string result;

    hipblasLtHandle_t handle;
    hipblasStatus_t status = hipblasLtCreate(&handle);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        return "hipblasLtCreate failed: " + std::to_string(status);
    }
    result += "hipblasLtCreate OK. ";

    // Check if FP4 types are available
    #ifdef HIPBLASLT_R_4F_E2M1
    result += "HIPBLASLT_R_4F_E2M1 defined. ";
    #else
    result += "HIPBLASLT_R_4F_E2M1 NOT defined. ";
    #endif

    #ifdef HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0
    result += "VEC32_UE8M0 defined. ";
    #else
    result += "VEC32_UE8M0 NOT defined. ";
    #endif

    hipblasLtDestroy(handle);
    return result;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
std::string probe_hipblaslt();
"""

_cb = os.path.expanduser("~/.cache/torch_extensions")
for d in os.listdir(_cb) if os.path.isdir(_cb) else []:
    cd = os.path.join(_cb, d, _MODULE)
    if os.path.isdir(cd): shutil.rmtree(cd, ignore_errors=True)

_mod = None
try:
    _mod = load_inline(name=_MODULE, cpp_sources=_CPP_SRC, cuda_sources=_HIP_SRC,
                       functions=['probe_hipblaslt'], verbose=True,
                       extra_cuda_cflags=['-O2', '-w', '--offload-arch=gfx950'],
                       extra_ldflags=['-lhipblaslt'])
    result = _mod.probe_hipblaslt()
    print(f"[255] hipBLASLt probe: {result}", file=sys.stderr)
except Exception as e:
    print(f"[255] hipBLASLt FAIL: {e}", file=sys.stderr)

_bc = [None, None, None]
@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]
    key = (B_shuffle.data_ptr(), B_scale_sh.data_ptr())
    if key != _bc[0]:
        _bc[0] = key
        _bc[1] = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        _bc[2] = B_scale_sh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
    return gemm_a16wfp4_preshuffle(A, _bc[1], _bc[2], prequant=True, dtype=torch.bfloat16)
