"""
MXFP4-MM: Hybrid Triton with tuned per-shape configs.
Uses shape-specific configs from aiter tuning + unshuffle for large K.
"""
from task import input_t, output_t
import torch
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4


def e8m0_unshuffle(scale, orig_m, orig_n):
    """Reverse e8m0_shuffle: inverse permute and unpad."""
    sm, sn = scale.shape
    scale = scale.view(sm // 32, sn // 8, 4, 16, 2, 2)
    scale = scale.permute(0, 5, 3, 1, 4, 2).contiguous()
    scale = scale.view(sm, sn)
    return scale[:orig_m, :orig_n]


# Shape-specific tuned configs from aiter config files
# Key: (N, K) -> {M_threshold: config}
TUNED_CONFIGS = {
    # N=2112, K=7168 (from gfx950-GEMM-AFP4WFP4-N=2112-K=7168.json)
    (2112, 7168): {
        16: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 256,
             "GROUP_SIZE_M": 1, "NUM_KSPLIT": 14,
             "num_warps": 4, "num_stages": 2, "waves_per_eu": 1,
             "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
    },
    # N=7168, K=2048 (from gfx950-GEMM-AFP4WFP4-N=7168-K=2048.json)
    (7168, 2048): {
        64: {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 1024,
             "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1,
             "num_warps": 4, "num_stages": 2, "waves_per_eu": 1,
             "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
    },
    # N=3072, K=1536 (from gfx950-GEMM-AFP4WFP4-N=3072-K=1536.json)
    (3072, 1536): {
        256: {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
              "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1,
              "num_warps": 4, "num_stages": 3, "waves_per_eu": 2,
              "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
    },
}


def _get_tuned_config(m, n, k):
    """Get shape-specific tuned config if available."""
    shape_configs = TUNED_CONFIGS.get((n, k))
    if shape_configs is None:
        return None
    # Find the best config for this M
    for m_thresh in sorted(shape_configs.keys()):
        if m <= m_thresh:
            return shape_configs[m_thresh]
    return None


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Quantize A
    A_q, A_scale = dynamic_mxfp4_quant(A)

    # Get tuned config for this shape
    config = _get_tuned_config(m, n, k)

    if k <= 512:
        # Small K: re-quantize B (cheaper than unshuffle)
        B_q_raw, B_scale = dynamic_mxfp4_quant(B)
        return gemm_afp4wfp4(A_q, B_q_raw, A_scale, B_scale,
                             dtype=torch.bfloat16, config=config)
    else:
        # Large K: unshuffle to avoid double quant
        B_q_uint8 = B_q.view(torch.uint8)
        B_scale = e8m0_unshuffle(B_scale_sh.view(torch.uint8), n, k // 32)
        return gemm_afp4wfp4(A_q, B_q_uint8, A_scale, B_scale,
                             dtype=torch.bfloat16, config=config)
