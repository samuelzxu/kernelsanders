"""
MLA decode - fused MXFP4 attention kernel using tl.dot_scaled.

Reads MXFP4 KV (fp4x2 + e8m0 scales) directly from kv_data["mxfp4"].
Uses MI355X native FP4 MFMA via tl.dot_scaled for Q@K^T computation.
2x bandwidth savings over fp8. Target: 13-20µs geomean.

Strategy B from README: quantize Q to e2m1 on-the-fly, compute QK^T
in fp4×fp4 using MFMA. Softmax in fp32. V from bf16.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from task import input_t, output_t

import aiter
from aiter import dtypes as aiter_dtypes
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op


BF16 = torch.bfloat16
FP32 = torch.float32
NH = 16
NKV = 1
DQ = 576
DV = 512
SM_SCALE = 1.0 / (DQ ** 0.5)
KV_LORA_RANK = 512
QK_ROPE_DIM = 64
SCALE_GROUP = 32
NUM_SCALE_BLOCKS = DQ // SCALE_GROUP  # 576/32 = 18


@triton.jit
def _mla_decode_mxfp4_stage1(
    Q,          # (bs, NH, DQ) bf16
    KV_fp4,     # (total_kv, 1, DQ//2) fp4x2 packed
    KV_scale,   # (total_kv_padded, NUM_SCALE_BLOCKS) fp8_e8m0
    V_bf16,     # (total_kv, 1, DV) bf16 — V from bf16 buffer
    Att_Out,    # (bs, NH, NUM_KV_SPLITS, KV_LORA_RANK+1) fp32
    kv_indptr,  # (bs+1,) int32
    kv_indices, # (total_kv,) int32
    stride_qb,
    stride_qh,
    stride_kv_fp4_token,   # stride between tokens in fp4 buffer
    stride_kv_scale_token, # stride between tokens in scale buffer
    stride_vb,             # stride between tokens in V bf16 buffer
    stride_att_b,
    stride_att_h,
    stride_att_s,
    sm_scale: tl.constexpr,
    kv_lora_rank: tl.constexpr,
    BLOCK_C: tl.constexpr,       # next_power_of_2(kv_lora_rank) = 512
    BLOCK_DQ_HALF: tl.constexpr, # DQ // 2 = 288 (for fp4x2 packed)
    BLOCK_N: tl.constexpr,       # KV tokens per tile (e.g., 32)
    BLOCK_H: tl.constexpr,       # query heads per tile (16)
    NUM_KV_SPLITS: tl.constexpr,
    NUM_SCALE_BLOCKS_CONST: tl.constexpr,  # 18
    batch: tl.constexpr,
):
    pid = tl.program_id(0)
    num_q_head_blk = tl.cdiv(NH, BLOCK_H)

    cur_head_id = pid % num_q_head_blk
    split_kv_id = (pid // num_q_head_blk) % NUM_KV_SPLITS
    cur_batch = pid // (num_q_head_blk * NUM_KV_SPLITS)

    if cur_batch >= batch:
        return

    cur_head = cur_head_id * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < NH

    # Load Q (bf16) for this batch and heads
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < kv_lora_rank
    offs_q_full = tl.arange(0, BLOCK_DQ_HALF * 2)  # 0..575 for full DQ
    mask_q_full = offs_q_full < DQ

    # Load full Q for score computation (576 dims)
    q_full = tl.load(
        Q + cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_q_full[None, :],
        mask=mask_h[:, None] & mask_q_full[None, :],
        other=0.0
    )  # (BLOCK_H, DQ) bf16

    # Quantize Q to MXFP4 for tl.dot_scaled
    q_mxfp4, q_scales = _mxfp4_quant_op(
        q_full.to(tl.float32), DQ, BLOCK_H, SCALE_GROUP
    )

    # KV split range
    cur_batch_kv_start = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    # Online softmax accumulators
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_C], dtype=tl.float32)

    # Offsets for fp4x2 K loading
    offs_k_half = tl.arange(0, BLOCK_DQ_HALF)  # 0..287 for packed fp4x2
    offs_scale = tl.arange(0, NUM_SCALE_BLOCKS_CONST)  # 0..17 for scale blocks

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)

            # Load KV page indices
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )

            # Load K from MXFP4 buffer: (BLOCK_N, DQ//2) fp4x2 packed
            k_fp4_ptrs = KV_fp4 + kv_loc[:, None] * stride_kv_fp4_token + offs_k_half[None, :]
            k_fp4 = tl.load(
                k_fp4_ptrs,
                mask=(offs_n[:, None] < split_kv_end) & (offs_k_half[None, :] < BLOCK_DQ_HALF),
                other=0,
            )  # (BLOCK_N, DQ//2) uint8/fp4x2

            # Load K scales: (BLOCK_N, NUM_SCALE_BLOCKS) fp8_e8m0
            k_scale_ptrs = KV_scale + kv_loc[:, None] * stride_kv_scale_token + offs_scale[None, :]
            k_scales = tl.load(
                k_scale_ptrs,
                mask=(offs_n[:, None] < split_kv_end) & (offs_scale[None, :] < NUM_SCALE_BLOCKS_CONST),
                other=0,
            )  # (BLOCK_N, 18)

            # Compute Q@K^T using hardware FP4 MFMA
            # q_mxfp4: (BLOCK_H, DQ//2) e2m1 packed
            # k_fp4: (BLOCK_N, DQ//2) e2m1 packed
            # q_scales: (BLOCK_H, NUM_SCALE_BLOCKS)
            # k_scales: (BLOCK_N, NUM_SCALE_BLOCKS)
            qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)
            qk = tl.dot_scaled(
                q_mxfp4, q_scales, "e2m1",
                k_fp4, k_scales, "e2m1",
                fast_math=True, acc=qk
            )
            qk *= sm_scale

            # Mask invalid positions
            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end),
                qk, float("-inf")
            )

            # Online softmax update
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]

            # Load V from bf16 buffer (first DV=512 dims)
            offs_v = tl.arange(0, BLOCK_C)
            v_ptrs = V_bf16 + kv_loc[:, None] * stride_vb + offs_v[None, :]
            v = tl.load(
                v_ptrs,
                mask=(offs_n[:, None] < split_kv_end) & (offs_v[None, :] < kv_lora_rank),
                other=0.0,
            )  # (BLOCK_N, DV) bf16

            # Accumulate weighted V
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

    # Store partial output (normalized V + log-sum-exp)
    offs_out_c = tl.arange(0, BLOCK_C)
    offs_mid_o = (
        cur_batch * stride_att_b
        + cur_head[:, None] * stride_att_h
        + split_kv_id * stride_att_s
        + offs_out_c[None, :]
    )
    tl.store(
        Att_Out + offs_mid_o,
        acc / tl.maximum(e_sum[:, None], 1e-12),
        mask=mask_h[:, None] & (offs_out_c[None, :] < kv_lora_rank),
    )

    # Store lse at position kv_lora_rank
    offs_lse = (
        cur_batch * stride_att_b
        + cur_head * stride_att_h
        + split_kv_id * stride_att_s
        + kv_lora_rank
    )
    tl.store(
        Att_Out + offs_lse,
        e_max + tl.log(tl.maximum(e_sum, 1e-12)),
        mask=mask_h,
    )


@triton.jit
def _mla_decode_mxfp4_stage2(
    logits,     # (bs, NH, NUM_KV_SPLITS, KV_LORA_RANK+1) fp32
    O,          # (bs, NH, DV) bf16
    kv_indptr,  # (bs+1,) int32
    stride_lb,
    stride_lh,
    stride_ls,
    stride_ob,
    stride_oh,
    Lv: tl.constexpr,
    head_num: tl.constexpr,
    batch: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """Reduce partial outputs from stage1 across KV splits."""
    pid = tl.program_id(0)
    cur_batch = pid // head_num
    cur_head = pid % head_num

    if cur_batch >= batch:
        return

    cur_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)
    kv_len_per_split = tl.cdiv(cur_seq_len, NUM_KV_SPLITS)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    # Reduce across splits using log-sum-exp
    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    for split_id in range(NUM_KV_SPLITS):
        # Check if this split has any tokens
        split_start = kv_len_per_split * split_id
        if split_start >= cur_seq_len:
            continue

        # Load partial V output
        offs_v = (
            cur_batch * stride_lb
            + cur_head * stride_lh
            + split_id * stride_ls
            + offs_d
        )
        tv = tl.load(logits + offs_v, mask=mask_d, other=0.0)

        # Load lse
        offs_lse = (
            cur_batch * stride_lb
            + cur_head * stride_lh
            + split_id * stride_ls
            + Lv  # lse is at position kv_lora_rank
        )
        tlogic = tl.load(logits + offs_lse)

        # Log-sum-exp reduction
        n_e_max = tl.maximum(tlogic, e_max)
        old_scale = tl.exp(e_max - n_e_max)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - n_e_max)
        acc += exp_logic * tv
        e_sum = e_sum * old_scale + exp_logic
        e_max = n_e_max

    # Write final output
    offs_o = cur_batch * stride_ob + cur_head * stride_oh + offs_d
    tl.store(
        O + offs_o,
        (acc / tl.maximum(e_sum, 1e-12)).to(O.dtype.element_ty),
        mask=mask_d,
    )


def mxfp4_fused_decode_attention(q, kv_fp4, kv_scale, kv_bf16, kv_indptr, bs, kv_seq_len):
    """Fused MXFP4 MLA decode attention using Triton with tl.dot_scaled."""
    n = kv_fp4.shape[0]
    nks = 16  # fewer splits since MXFP4 reads less data

    # Pre-allocate
    kvi = torch.arange(n, dtype=torch.int32, device="cuda")
    o = torch.empty((bs, NH, DV), dtype=BF16, device="cuda")
    att_out = torch.empty((bs, NH, nks, KV_LORA_RANK + 1), dtype=FP32, device="cuda")

    # Flatten fp4 and scale tensors
    kv_fp4_flat = kv_fp4.view(n, DQ // 2)  # (n, 288)
    kv_scale_flat = kv_scale  # (n_padded, NUM_SCALE_BLOCKS)
    v_bf16_flat = kv_bf16.view(n, DQ)[:, :DV]  # (n, 512) bf16 — V from bf16 buffer

    BLOCK_N = 32
    BLOCK_H = 16
    BLOCK_C = 512  # next_power_of_2(KV_LORA_RANK)
    BLOCK_DQ_HALF = DQ // 2  # 288
    grid = (bs * (NH // BLOCK_H) * nks,)

    _mla_decode_mxfp4_stage1[grid](
        q, kv_fp4_flat, kv_scale_flat, v_bf16_flat, att_out,
        kv_indptr, kvi,
        q.stride(0), q.stride(1),
        kv_fp4_flat.stride(0),
        kv_scale_flat.stride(0),
        v_bf16_flat.stride(0),
        att_out.stride(0), att_out.stride(1), att_out.stride(2),
        sm_scale=SM_SCALE,
        kv_lora_rank=KV_LORA_RANK,
        BLOCK_C=BLOCK_C,
        BLOCK_DQ_HALF=BLOCK_DQ_HALF,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=nks,
        NUM_SCALE_BLOCKS_CONST=NUM_SCALE_BLOCKS,
        batch=bs,
        num_warps=8,
        num_stages=2,
    )

    # Stage 2: reduce across KV splits
    BLOCK_DV = triton.next_power_of_2(DV)
    grid2 = (bs * NH,)

    _mla_decode_mxfp4_stage2[grid2](
        att_out, o, kv_indptr,
        att_out.stride(0), att_out.stride(1), att_out.stride(2),
        o.stride(0), o.stride(1),
        Lv=KV_LORA_RANK,
        head_num=NH,
        batch=bs,
        NUM_KV_SPLITS=nks,
        BLOCK_DV=BLOCK_DV,
        num_warps=4,
        num_stages=2,
    )

    return o


# Compiled GEMM for small configs (unchanged)
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

    # GEMM for small configs
    if bs <= 4 or (bs <= 32 and kv_seq_len <= 1024):
        kv = kv_data["bf16"].view(bs, kv_seq_len, DQ)
        q_3d = q.view(bs, NH, DQ)
        v = kv[:, :, :DV]
        kv_t = kv.transpose(-2, -1)
        if kv_seq_len <= 1024:
            return _compiled_gemm_short(q_3d, kv_t, v)
        else:
            return _compiled_gemm_long(q_3d, kv_t, v)

    # Fused MXFP4 attention for larger configs
    kv_fp4, kv_scale = kv_data["mxfp4"]
    kv_bf16 = kv_data["bf16"]

    return mxfp4_fused_decode_attention(
        q, kv_fp4, kv_scale, kv_bf16, kv_indptr, bs, kv_seq_len
    )
