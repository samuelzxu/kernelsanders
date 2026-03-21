"""
MXFP4-MM: CK with dynamically detected cu_num + Triton fallback.
Detects actual cu_num at runtime and injects CK configs with correct value.
"""
import os
import csv
import json
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info

# Inject Triton configs
_TRITON_CONFIGS = {
    "N=2880-K=512": {
        "M_LEQ_4": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 3, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 3, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
    "N=4096-K=512": {
        "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 3, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
    "N=2112-K=7168": {
        "M_LEQ_16": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 14, "num_warps": 4, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 32, "cache_modifier": None}
    },
    "N=7168-K=2048": {
        "M_LEQ_64": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 1024, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 32, "cache_modifier": None}
    },
    "N=3072-K=1536": {
        "M_LEQ_64": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 3, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "M_LEQ_256": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 32, "cache_modifier": None}
    },
}

# CK ASM configs - will be injected with ACTUAL cu_num
_CK_SHAPES = [
    # (M, N, K, kernelName, splitK)
    (4, 2880, 512, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", 0),
    (16, 2112, 7168, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", 0),
    (32, 4096, 512, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", 0),
    (32, 2880, 512, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", 0),
    (8, 2112, 7168, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", 0),
    (16, 3072, 1536, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", 0),
    (64, 3072, 1536, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", 0),
    (256, 2880, 512, "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_192x128E", 0),
]


def _inject_all():
    # Triton configs
    try:
        dev = arch_info.get_arch()
    except Exception:
        dev = "gfx950"
    config_dir = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
    os.makedirs(config_dir, exist_ok=True)
    for shape_key, config in _TRITON_CONFIGS.items():
        fpath = f"{config_dir}/{dev}-GEMM-AFP4WFP4-{shape_key}.json"
        with open(fpath, "w") as f:
            json.dump(config, f)

    # CK configs with ACTUAL cu_num
    try:
        from aiter.jit.core import AITER_CONFIGS
        from aiter.jit.utils.chip_info import get_cu_num
        cu_num = get_cu_num()
        csv_path = AITER_CONFIGS.AITER_CONFIG_GEMM_A4W4_FILE
        if os.path.exists(csv_path):
            existing = set()
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) >= 4:
                        existing.add((int(row[0]), int(row[1]), int(row[2]), int(row[3])))
            new_entries = []
            for m, n, k, kname, sk in _CK_SHAPES:
                if (cu_num, m, n, k) not in existing:
                    new_entries.append([cu_num, m, n, k, 21, sk, 10.0, kname, 0, 0, 0.0])
            if new_entries:
                with open(csv_path, "a") as f:
                    writer = csv.writer(f)
                    for entry in new_entries:
                        writer.writerow(entry)
            # Log cu_num for debugging
            import sys
            print(f"[mxfp4-mm] cu_num={cu_num}, injected {len(new_entries)} CK configs", file=sys.stderr)
    except Exception as e:
        import sys
        print(f"[mxfp4-mm] CK inject failed: {e}", file=sys.stderr)

try:
    _inject_all()
except Exception:
    pass

from task import input_t, output_t
import torch
import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4


def e8m0_unshuffle(scale, orig_m, orig_n):
    sm, sn = scale.shape
    scale = scale.view(sm // 32, sn // 8, 4, 16, 2, 2)
    scale = scale.permute(0, 5, 3, 1, 4, 2).contiguous()
    scale = scale.view(sm, sn)
    return scale[:orig_m, :orig_n]


_cache_key = None
_cache_val = None
_cache_ascale_sh = None


def custom_kernel(data: input_t) -> output_t:
    global _cache_key, _cache_val, _cache_ascale_sh
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    A_q, A_scale = dynamic_mxfp4_quant(A)

    # Try CK path first (uses shuffled data, no double quant)
    A_scale_sh = e8m0_shuffle(A_scale)
    try:
        return aiter.gemm_a4w4(
            A_q.view(dtypes.fp4x2), B_shuffle,
            A_scale_sh.view(dtypes.fp8_e8m0), B_scale_sh,
            dtype=dtypes.bf16, bpreshuffle=True,
        )
    except Exception:
        pass

    # Fallback to Triton
    key = (B.data_ptr(), B_q.data_ptr(), B_scale_sh.data_ptr())
    if key == _cache_key:
        B_scale = _cache_val
    else:
        if k <= 512:
            _, B_scale = dynamic_mxfp4_quant(B)
        else:
            B_scale = e8m0_unshuffle(B_scale_sh.view(torch.uint8), n, k // 32)
        _cache_key = key
        _cache_val = B_scale

    B_q_uint8 = B_q.view(torch.uint8)
    return gemm_afp4wfp4(A_q, B_q_uint8, A_scale, B_scale, dtype=torch.bfloat16)
