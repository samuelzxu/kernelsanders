"""
MLA decode - GEMM for small cases, assembly for the rest.

Key optimization: adaptive nks targeting ~256 total work items (1 full wave on MI355X).
- MI355X has 256 CUs; 256 work items = 1 wave with zero tail effect
- Fewer splits = smaller lg/ls reduce buffers = less data movement in reduce step
- For bs=256/nks=32: lg=268MB. For bs=256/nks=4: lg=34MB (8x smaller)

Changes vs 137:
- kv>1024: nks = max(4, 256//bs) → 8 for bs=32, 4 for bs=64, 4 for bs=256
- kv<=1024: nks=4 (was 8) → 1 wave for bs=64, 4 waves for bs=256
"""

import torch
import torch.nn.functional as F
from task import input_t, output_t

import aiter
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

    kv_buffer, kv_scale = kv_data["fp8"]
    n = kv_buffer.shape[0]

    if kv_seq_len <= 1024:
        # a16w8 path: bf16 Q + fp8 KV, no Q quantization needed
        # nks=4: bs=64→256 CTAs=1 wave, bs=256→1024 CTAs=4 waves (more manageable)
        q_input, q_scale, qd = q, None, BF16
        kvg, nks = 16, 4
    else:
        # a8w8 path: fp8 Q + fp8 KV
        # Adaptive nks targeting ~256 total work items (1 full wave on MI355X 256 CUs)
        # Fewer splits → smaller lg/ls buffers → faster reduce step
        q_input, q_scale = quantize_fp8_triton(q)
        qd, kvg = FP8_DTYPE, 64
        # max(4, 256//bs): floor 4 to avoid extreme work-per-CTA
        # bs=32: nks=8 (256 CTAs=1 wave)
        # bs=64: nks=4 (256 CTAs=1 wave)
        # bs=256: nks=4 (1024 CTAs=4 waves, conservative)
        nks = max(4, 256 // bs)

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
