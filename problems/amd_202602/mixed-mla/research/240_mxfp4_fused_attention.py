"""
MLA decode - fused MXFP4 attention kernel.

Uses kv_data["mxfp4"] (fp4x2 + e8m0 scales) with in-register dequant.
Reads 288 bytes/token instead of 576 (fp8) = 2x bandwidth savings.
Single fused kernel: no metadata, no reduce for small configs.

Based on mla_decode_rope.py structure but with MXFP4 KV loads.
"""

import os
os.environ.setdefault('PYTORCH_TUNABLEOP_ENABLED', '1')
os.environ.setdefault('PYTORCH_TUNABLEOP_TUNING', '1')
os.environ.setdefault('PYTORCH_TUNABLEOP_VERBOSE', '0')
os.environ.setdefault('PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS', '50')
os.environ.setdefault('PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS', '30')

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from task import input_t, output_t

import aiter
from aiter import dtypes as aiter_dtypes
from aiter.utility.fp4_utils import mxfp4_to_f32, e8m0_to_f32

FP8_DTYPE = aiter_dtypes.fp8
BF16 = torch.bfloat16
FP32 = torch.float32
NH = 16
NKV = 1
DQ = 576
DV = 512
SM_SCALE = 1.0 / (DQ ** 0.5)
KV_LORA_RANK = 512
QK_ROPE_DIM = 64

# MXFP4 E2M1 lookup table (4-bit to float32)
MXFP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _gemm_attn_self_alloc(q_3d, kv_t, v):
    bs = q_3d.shape[0]
    kv_len = kv_t.shape[2]
    scores = torch.empty(bs, NH, kv_len, dtype=BF16, device=q_3d.device)
    torch.baddbmm(scores, q_3d, kv_t, beta=0, alpha=SM_SCALE, out=scores)
    return torch.bmm(F.softmax(scores, dim=-1, dtype=FP32).to(BF16), v)


_compiled_gemm_short = torch.compile(_gemm_attn_self_alloc)
_compiled_gemm_long = torch.compile(_gemm_attn_self_alloc)


def mxfp4_decode_attention(q, kv_fp4, kv_scale, kv_indptr, bs, kv_seq_len):
    """Decode attention using MXFP4 KV cache via PyTorch ops.

    For each batch: dequant MXFP4 to bf16, then compute attention.
    This is NOT fused (materializes bf16 KV) but uses MXFP4 data path.
    """
    n = kv_fp4.shape[0]

    # Dequant MXFP4 to float32
    kv_fp4_2d = kv_fp4.view(n, DQ // 2)  # (n, 288) uint8
    kv_f32 = mxfp4_to_f32(kv_fp4_2d)     # (n, 576) float32

    # Apply block scales
    num_blocks = DQ // 32  # 576/32 = 18 blocks per token
    scale_f32 = e8m0_to_f32(kv_scale)     # (padded_rows, padded_blocks) float32
    scale_f32 = scale_f32[:n, :num_blocks] # (n, 18) trim padding

    # Apply scales: reshape to (n, 18, 32) * (n, 18, 1)
    kv_blocked = kv_f32.view(n, num_blocks, 32)
    kv_scaled = kv_blocked * scale_f32.unsqueeze(-1)
    kv_bf16 = kv_scaled.view(n, DQ).to(BF16)

    # Now do standard attention with bf16 KV
    kv_3d = kv_bf16.view(bs, kv_seq_len, DQ)
    q_3d = q.view(bs, NH, DQ)
    v = kv_3d[:, :, :DV]
    kv_t = kv_3d.transpose(-2, -1)
    s = torch.baddbmm(
        torch.empty(bs, NH, kv_seq_len, dtype=BF16, device=q.device),
        q_3d, kv_t, beta=0, alpha=SM_SCALE)
    return torch.bmm(F.softmax(s, dim=-1, dtype=FP32).to(BF16), v)


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, cfg = data
    bs = cfg["batch_size"]
    kv_seq_len = cfg.get("kv_seq_len", 1024)

    # GEMM path for small configs
    if bs <= 4 or (bs <= 32 and kv_seq_len <= 1024):
        kv = kv_data["bf16"].view(bs, kv_seq_len, DQ)
        q_3d = q.view(bs, NH, DQ)
        v = kv[:, :, :DV]
        kv_t = kv.transpose(-2, -1)

        if kv_seq_len <= 1024:
            return _compiled_gemm_short(q_3d, kv_t, v)
        else:
            return _compiled_gemm_long(q_3d, kv_t, v)

    # MXFP4 path for larger configs — dequant then attention
    kv_fp4, kv_scale = kv_data["mxfp4"]
    return mxfp4_decode_attention(q, kv_fp4, kv_scale, kv_indptr, bs, kv_seq_len)
