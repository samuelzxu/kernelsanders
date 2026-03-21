"""
MLA decode - GEMM for small cases, mla_decode_fwd for the rest.

Key change: use mla_decode_fwd (high-level wrapper from aiter.mla)
instead of manually calling stage1_asm_fwd + reduce_v1.
mla_decode_fwd may handle lg/ls allocation more efficiently internally
(single allocation, pre-allocated pool, etc.), saving 2-3µs per call.

Also: use our kvg settings (kvg=64 for kv>1024) with the high-level API.
"""

import torch
import torch.nn.functional as F
from task import input_t, output_t

from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.ops.triton.quant.quant import dynamic_per_tensor_quant_fp8_i8

FP8_DTYPE = aiter_dtypes.fp8
BF16 = torch.bfloat16
FP32 = torch.float32
NH = 16
NKV = 1
DQ = 576
DV = 512
SM_SCALE = 1.0 / (DQ ** 0.5)


def quantize_fp8_triton(tensor):
    t2d = tensor.view(-1, tensor.shape[-1])
    qx = torch.empty_like(t2d, dtype=FP8_DTYPE)
    scale = torch.zeros(1, dtype=FP32, device=tensor.device)
    dynamic_per_tensor_quant_fp8_i8(qx, t2d, scale)
    return qx.view(tensor.shape), scale


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

    # Assembly path via mla_decode_fwd (high-level wrapper)
    kv_buffer, kv_scale = kv_data["fp8"]
    n = kv_buffer.shape[0]

    if kv_seq_len <= 1024:
        q_input, q_scale, qd = q, None, BF16
        kvg, nks = 16, 8
    else:
        q_input, q_scale = quantize_fp8_triton(q)
        qd, kvg = FP8_DTYPE, 64
        nks = 32

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

    o = torch.empty((bs, NH, DV), dtype=BF16, device="cuda")
    kv_buffer_4d = kv_buffer.view(n, 1, NKV, DQ)

    mla_decode_fwd(
        q_input.view(-1, NH, DQ),
        kv_buffer_4d,
        o,
        qo_indptr, kv_indptr, kvi, klp,
        1,  # max_seqlen_q
        page_size=1,
        nhead_kv=NKV,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        num_kv_splits=nks,
        q_scale=q_scale,
        kv_scale=kv_scale,
        intra_batch_mode=True,
        work_meta_data=wm,
        work_indptr=wi,
        work_info_set=wis,
        reduce_indptr=ri,
        reduce_final_map=rfm,
        reduce_partial_map=rpm,
    )
    return o
