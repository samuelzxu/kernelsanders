#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#312: Check e8m0_shuffle for inverse parameter. Also check e8m0_unshuffle.
"""
import torch, sys, inspect
from task import input_t, output_t

print("=== e8m0_shuffle analysis ===")
import aiter
shuffle_fn = aiter.fp4_utils.e8m0_shuffle

# Check if it's a Triton kernel with source
fn = shuffle_fn
for _ in range(3):
    if hasattr(fn, 'fn'): fn = fn.fn
try:
    src = inspect.getsource(fn)
    print(f"SOURCE ({len(src)} chars):")
    print(src[:3000])
except Exception as e:
    print(f"No source: {e}, type={type(fn)}")

# Check for unshuffle
print("\n=== Unshuffle search ===")
fp4 = aiter.fp4_utils
all_attrs = [x for x in dir(fp4) if 'shuffle' in x.lower() or 'unshuffle' in x.lower()]
print(f"Shuffle-related attrs: {all_attrs}")

# Check the function signature and possible kwargs
print(f"\nFunction type: {type(shuffle_fn)}")
try:
    sig = inspect.signature(shuffle_fn)
    print(f"Signature: {sig}")
except: print("No signature available")

# Try calling with different params
scale = torch.arange(48, dtype=torch.uint8, device="cuda").unsqueeze(0).expand(32, -1).contiguous()
print(f"\nInput: {scale.shape}, {scale[0,:8].tolist()}")

# Normal shuffle
try:
    r = shuffle_fn(scale)
    print(f"shuffle(): {r.shape}, {r[0,:8].tolist()}")
except Exception as e:
    print(f"shuffle() error: {e}")

# Try with inverse=True
try:
    r = shuffle_fn(scale, inverse=True)
    print(f"shuffle(inverse=True): {r.shape}")
except Exception as e:
    print(f"inverse=True: {e}")

# Try with unshuffle=True
try:
    r = shuffle_fn(scale, unshuffle=True)
    print(f"shuffle(unshuffle=True): {r.shape}")
except Exception as e:
    print(f"unshuffle=True: {e}")

# Also try the dynamic_mxfp4_quant with shuffle param
print("\n=== dynamic_mxfp4_quant shuffle ===")
from aiter.ops.triton.quant import dynamic_mxfp4_quant
B = torch.randn(32, 1536, dtype=torch.bfloat16, device="cuda")
B_q, B_scale = dynamic_mxfp4_quant(B)
print(f"No shuffle: B_scale shape={B_scale.shape}, strides={B_scale.stride()}")

try:
    B_q2, B_scale2 = dynamic_mxfp4_quant(B, shuffle=True)
    print(f"shuffle=True: B_scale2 shape={B_scale2.shape}")
except TypeError as e:
    print(f"shuffle param not supported: {e}")

# Fallback kernel
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
import os, json
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
_cfgs = {"N=2880-K=512": {"M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=4096-K=512": {"M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=2112-K=7168": {"M_LEQ_16": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 8, "num_warps": 4, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=7168-K=2048": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=3072-K=1536": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 3, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}}
try: _dev = arch_info.get_arch()
except: _dev = "gfx950"
_cd = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
os.makedirs(_cd, exist_ok=True)
for _sk, _cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json", "w") as f:
        json.dump(_cfg, f)

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
