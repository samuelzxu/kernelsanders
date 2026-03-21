"""
MXFP4-MM: Submission template for pre-compiled gluon kernel.

Architecture:
  K=512, K=7168: gemm_a16wfp4_preshuffle (fused, already fast)
  K=2048, K=1536: pre-compiled gluon GEMM via hipModuleLoad

After running compile_gluon.py on AMD Developer Cloud, paste the base64
binary below and the kernel function name from the .hsaco metadata.

USAGE:
  1. Run compile_gluon.py on gfx950 machine
  2. Copy the .b64 file contents into _GLUON_K1536_B64 and _GLUON_K2048_B64
  3. Update _GLUON_K1536_FN and _GLUON_K2048_FN with the kernel function names
  4. Submit this file
"""
import os, json, base64, tempfile, ctypes
import torch
from task import input_t, output_t
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle

# ============================================================
# PRE-COMPILED GLUON BINARIES (paste from compile_gluon.py output)
# ============================================================

# K=1536 M=256 gluon kernel binary (base64)
_GLUON_K1536_B64 = ""  # TODO: paste base64 here

# K=2048 M=64 gluon kernel binary (base64)
_GLUON_K2048_B64 = ""  # TODO: paste base64 here

# Kernel function names (from compilation output)
_GLUON_K1536_FN = b""  # TODO: e.g. b"_gemm_afp4wfp4_kernel_..."
_GLUON_K2048_FN = b""  # TODO

# ============================================================
# PRESHUFFLE CONFIGS (for K=512/K=7168 shapes)
# ============================================================

_PS_CONFIGS = {
    "N=2880-K=512": {
        "M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
    "N=4096-K=512": {
        "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
}

def _inject_configs():
    try:
        dev = arch_info.get_arch()
    except Exception:
        dev = "gfx950"
    config_dir = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
    os.makedirs(config_dir, exist_ok=True)
    for shape_key, config in _PS_CONFIGS.items():
        fpath = f"{config_dir}/{dev}-GEMM-A16WFP4_PRESHUFFLED-{shape_key}.json"
        with open(fpath, "w") as f:
            json.dump(config, f)

_inject_configs()

# ============================================================
# LOAD PRE-COMPILED GLUON BINARIES VIA HIP RUNTIME
# ============================================================

_hip = None
_gluon_kernels = {}  # k -> (hipFunction, tileM, tileN, config)

def _load_gluon_binary(b64_str, fn_name, tile_m, tile_n, config, key):
    """Load a pre-compiled .hsaco binary and register the kernel."""
    global _hip
    if not b64_str:
        return False

    if _hip is None:
        _hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')

    binary = base64.b64decode(b64_str)
    with tempfile.NamedTemporaryFile(suffix='.hsaco', delete=False) as f:
        f.write(binary)
        path = f.name

    module = ctypes.c_void_p()
    rc = _hip.hipModuleLoad(ctypes.byref(module), path.encode())
    os.unlink(path)

    if rc != 0:
        return False

    func = ctypes.c_void_p()
    rc = _hip.hipModuleGetFunction(ctypes.byref(func), module, fn_name)
    if rc != 0:
        return False

    _gluon_kernels[key] = (func, tile_m, tile_n, config)
    return True

# Try loading pre-compiled kernels
_has_gluon_k1536 = _load_gluon_binary(
    _GLUON_K1536_B64, _GLUON_K1536_FN, 256, 256,
    {"BSM": 256, "BSN": 256, "BSK": 256, "KSPLIT": 1}, 1536
)
_has_gluon_k2048 = _load_gluon_binary(
    _GLUON_K2048_B64, _GLUON_K2048_FN, 64, 256,
    {"BSM": 64, "BSN": 256, "BSK": 256, "KSPLIT": 1}, 2048
)


def e8m0_unshuffle(scale, orig_m, orig_n):
    sm, sn = scale.shape
    scale = scale.view(sm // 32, sn // 8, 4, 16, 2, 2)
    scale = scale.permute(0, 5, 3, 1, 4, 2).contiguous()
    scale = scale.view(sm, sn)
    return scale[:orig_m, :orig_n]


# Caches
_ps_cache_key = None
_ps_cache_w = None
_ps_cache_ws = None
_gluon_bscale_key = None
_gluon_bscale_val = None


def _launch_gluon(func, A_q, B_q_t, A_scale, B_scale, out, m, n, k_packed, tile_m, tile_n):
    """Launch pre-compiled gluon kernel via hipModuleLaunchKernel."""
    # TODO: Set up kernel args struct and launch
    # This requires matching the exact argument layout of the compiled kernel
    # The gluon kernel args: a_ptr, b_ptr, c_ptr, a_scales_ptr, b_scales_ptr,
    #   M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
    #   stride_ck, stride_cm, stride_cn, stride_asm, stride_ask,
    #   stride_bsn, stride_bsk
    pass


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _ps_cache_key, _ps_cache_w, _ps_cache_ws
    global _gluon_bscale_key, _gluon_bscale_val
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # K=2048/K=1536 with pre-compiled gluon: quant(A) + gluon GEMM
    if k in _gluon_kernels:
        B_q_uint8 = B_q.view(torch.uint8)
        key = (B.data_ptr(), B_q.data_ptr(), B_scale_sh.data_ptr())
        if key == _gluon_bscale_key:
            B_scale = _gluon_bscale_val
        else:
            B_scale = e8m0_unshuffle(B_scale_sh.view(torch.uint8), n, k // 32)
            _gluon_bscale_key = key
            _gluon_bscale_val = B_scale

        A_q, A_scale = dynamic_mxfp4_quant(A)
        # TODO: Launch gluon kernel
        # For now, fall through to preshuffle
        pass

    # K=512, K=7168 (and fallback): preshuffle
    key = (B_shuffle.data_ptr(), B_scale_sh.data_ptr())
    if key == _ps_cache_key:
        B_w = _ps_cache_w
        B_ws = _ps_cache_ws
    else:
        B_w = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        B_ws = B_scale_sh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
        _ps_cache_key = key
        _ps_cache_w = B_w
        _ps_cache_ws = B_ws

    return gemm_a16wfp4_preshuffle(A, B_w, B_ws, prequant=True, dtype=torch.bfloat16)
