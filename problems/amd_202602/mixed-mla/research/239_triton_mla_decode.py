"""
MLA decode - use aiter's Triton MLA decode kernel instead of assembly.

NOVEL APPROACH: aiter has a Triton MLA decode kernel specifically designed
for DeepSeek's 512+64 dim split (mla_decode_rope.py). It handles the
nope (512) + rope (64) dot products separately, which is more efficient
than treating them as a single 576-dim dot product.

Pre-configured for gfx950 (MI355X): BLOCK_N=32, BLOCK_H=16, 8 warps.

For absorbed MLA (our case): use_rope=False (RoPE already absorbed into Q/K).
The kernel still splits the computation into 512+64 dim parts efficiently.

Advantages over assembly kernel:
- Purpose-built for MLA's latent dimension structure
- No metadata computation needed
- Triton auto-optimized for the specific shape
- Two-stage split-K with built-in reduction (no separate reduce call)
"""

import os
os.environ.setdefault('PYTORCH_TUNABLEOP_ENABLED', '1')
os.environ.setdefault('PYTORCH_TUNABLEOP_TUNING', '1')
os.environ.setdefault('PYTORCH_TUNABLEOP_VERBOSE', '0')
os.environ.setdefault('PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS', '50')
os.environ.setdefault('PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS', '30')

import torch
import torch.nn.functional as F
from task import input_t, output_t

import aiter
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.ops.triton.quant.quant import dynamic_per_tensor_quant_fp8_i8
from aiter.ops.triton.attention.mla_decode_rope import decode_attention_fwd_grouped_rope

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


def _gemm_attn_self_alloc(q_3d, kv_t, v):
    bs = q_3d.shape[0]
    kv_len = kv_t.shape[2]
    scores = torch.empty(bs, NH, kv_len, dtype=BF16, device=q_3d.device)
    torch.baddbmm(scores, q_3d, kv_t, beta=0, alpha=SM_SCALE, out=scores)
    return torch.bmm(F.softmax(scores, dim=-1, dtype=FP32).to(BF16), v)


_compiled_gemm_short = torch.compile(_gemm_attn_self_alloc)
_compiled_gemm_long = torch.compile(_gemm_attn_self_alloc)


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, cfg = data
    bs = cfg["batch_size"]
    kv_seq_len = cfg.get("kv_seq_len", 1024)

    # GEMM path for small configs (unchanged)
    if bs <= 4 or (bs <= 32 and kv_seq_len <= 1024):
        kv = kv_data["bf16"].view(bs, kv_seq_len, DQ)
        q_3d = q.view(bs, NH, DQ)
        v = kv[:, :, :DV]
        kv_t = kv.transpose(-2, -1)

        if kv_seq_len <= 1024:
            return _compiled_gemm_short(q_3d, kv_t, v)
        else:
            return _compiled_gemm_long(q_3d, kv_t, v)

    # Triton MLA decode kernel for assembly configs
    kv_bf16 = kv_data["bf16"]  # (n, 1, 576) bf16
    n = kv_bf16.shape[0]
    nks = 32

    q_3d = q.view(bs, NH, DQ)
    kvi = torch.arange(n, dtype=torch.int32, device="cuda")
    o = torch.empty((bs, NH, DV), dtype=BF16, device="cuda")

    # Intermediate logits: (bs, NH, nks, KV_LORA_RANK + 1) float32
    # Layout: batch × head × split × (v_partial[512] + lse[1])
    attn_logits = torch.empty(
        (bs, NH, nks, KV_LORA_RANK + 1), dtype=FP32, device="cuda"
    )

    # Dummy tensors for unused rope parameters
    dummy = torch.empty(0, device="cuda")

    decode_attention_fwd_grouped_rope(
        q=q_3d,
        k_buffer=kv_bf16,          # (n, 1, 576) — full K with nope+rope
        v_buffer=kv_bf16[:, :, :DV],  # (n, 1, 512) — V = first 512 dims
        o=o,
        kv_indptr=kv_indptr,
        kv_indices=kvi,
        k_pe_tokens=None,          # no rope output needed
        kv_lora_rank=KV_LORA_RANK,
        rotary_dim=QK_ROPE_DIM,
        cos_sin_cache=dummy,       # not used when use_rope=False
        positions=dummy,           # not used when use_rope=False
        attn_logits=attn_logits,
        num_kv_splits=nks,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        use_rope=False,            # absorbed MLA — rope already in Q/K
        is_neox_style=False,
    )
    return o
