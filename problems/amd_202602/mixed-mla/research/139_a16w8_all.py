"""
MLA decode - GEMM for small cases, a16w8 assembly for the rest.

Key: use a16w8 (bf16 Q + fp8 KV) for ALL assembly cases, including kv>1024.
This eliminates the Q quantization overhead (~3-5µs per call).

Why this might work:
- Q data is tiny vs KV: bs=64 → Q=1.2MB bf16, KV=301MB fp8 (250:1 ratio)
- The a16w8 kernel operates on bf16 Q + fp8 KV — should work for any kv length
- Savings: skip Triton quantization kernel launch + memory copy

Changes vs 137:
- kv>1024 path now uses q_input=q (bf16), q_scale=None, qd=BF16
- kvg stays at 64 (appropriate for long sequences)
- nks stays at 32 (unchanged)
"""

import torch
import torch.nn.functional as F
from task import input_t, output_t

import aiter
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

FP8_DTYPE = aiter_dtypes.fp8
BF16 = torch.bfloat16
FP32 = torch.float32
NH = 16
NKV = 1
DQ = 576
DV = 512
SM_SCALE = 1.0 / (DQ ** 0.5)


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, cfg = data
    bs = cfg["batch_size"]
    kv_seq_len = cfg.get("kv_seq_len", 1024)

    # GEMM path: bs<=4 (all kv) or bs<=32 with short kv
    if bs <= 4 or (bs <= 32 and kv_seq_len <= 1024):
        kv = kv_data["bf16"].view(bs, kv_seq_len, DQ)
        q_3d = q.view(bs, NH, DQ)
        v = kv[:, :, :DV]
        kv_t = kv.transpose(-2, -1)
        s = torch.baddbmm(
            torch.empty(bs, NH, kv_seq_len, dtype=BF16, device=q.device),
            q_3d, kv_t, beta=0, alpha=SM_SCALE)
        return torch.bmm(F.softmax(s, dim=-1, dtype=FP32).to(BF16), v)

    kv_buffer, kv_scale = kv_data["fp8"]
    n = kv_buffer.shape[0]

    # Always use a16w8: bf16 Q + fp8 KV, no Q quantization needed
    # For kv<=1024: kvg=16, nks=8 (standard config)
    # For kv>1024: kvg=64, nks=32 (same as before but no Q quant)
    q_input, q_scale, qd = q, None, BF16
    if kv_seq_len <= 1024:
        kvg, nks = 16, 8
    else:
        kvg, nks = 64, 32

    kvd = kv_buffer.dtype
    info = get_mla_metadata_info_v1(
        bs, 1, NH, qd, kvd, is_sparse=False, fast_mode=False,
        num_kv_splits=nks, intra_batch_mode=True)
    wm, wi, wis, ri, rfm, rpm = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    klp = qo_indptr[1:bs+1]
    kvi = torch.arange(n, dtype=torch.int32, device="cuda")

    get_mla_metadata_v1(
        qo_indptr, kv_indptr, klp, NH // NKV, NKV, True,
        wm, wis, wi, ri, rfm, rpm,
        page_size=1, kv_granularity=kvg,
        max_seqlen_qo=1, uni_seqlen_qo=1,
        fast_mode=False, max_split_per_batch=nks,
        intra_batch_mode=True, dtype_q=qd, dtype_kv=kvd)

    np_ = bs * nks
    lg = torch.empty((np_, 1, NH, DV), dtype=FP32, device="cuda")
    ls = torch.empty((np_, 1, NH, 1), dtype=FP32, device="cuda")
    o = torch.empty((bs, NH, DV), dtype=BF16, device="cuda")

    aiter.mla_decode_stage1_asm_fwd(
        q_input.view(-1, NH, DQ),
        kv_buffer.view(n, 1, NKV, DQ),
        qo_indptr, kv_indptr, kvi, klp,
        None, wm, wi, wis, 1, 1, NKV, SM_SCALE,
        lg, ls, o, q_scale, kv_scale)
    aiter.mla_reduce_v1(lg, ls, ri, rfm, rpm, 1, o, None)
    return o
