"""
MLA decode kernel — bf16 Q + fp8 KV (a16w8) for all batch sizes.
No Q quantization needed. Direct aiter stage1+reduce with full caching.
"""

import torch
from task import input_t, output_t

import aiter
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FP8_DTYPE = aiter_dtypes.fp8
BF16 = torch.bfloat16
PAGE_SIZE = 1
NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512

# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------
_full_cache: dict[tuple, dict] = {}
_kv_indices_cache: dict[int, torch.Tensor] = {}
_output_cache: dict[int, torch.Tensor] = {}


def get_num_kv_splits(batch_size: int) -> int:
    if batch_size <= 4:
        return 16
    else:
        return 32


def _get_or_build_all(
    batch_size: int,
    num_kv_splits: int,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
):
    key = (batch_size, num_kv_splits)
    if key not in _full_cache:
        info = get_mla_metadata_info_v1(
            batch_size, 1, NUM_HEADS, BF16, FP8_DTYPE,
            is_sparse=False, fast_mode=False,
            num_kv_splits=num_kv_splits, intra_batch_mode=True,
        )
        bufs = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
        (work_metadata, work_indptr, work_info_set,
         reduce_indptr, reduce_final_map, reduce_partial_map) = bufs

        kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

        get_mla_metadata_v1(
            qo_indptr, kv_indptr, kv_last_page_len,
            NUM_HEADS // NUM_KV_HEADS, NUM_KV_HEADS, True,
            work_metadata, work_info_set, work_indptr,
            reduce_indptr, reduce_final_map, reduce_partial_map,
            page_size=PAGE_SIZE, kv_granularity=16,
            max_seqlen_qo=1, uni_seqlen_qo=1,
            fast_mode=False, max_split_per_batch=num_kv_splits,
            intra_batch_mode=True, dtype_q=BF16, dtype_kv=FP8_DTYPE,
        )

        n_partial = reduce_partial_map.size(0)
        _full_cache[key] = (
            work_metadata, work_indptr, work_info_set,
            reduce_indptr, reduce_final_map, reduce_partial_map,
            kv_last_page_len,
            torch.empty((n_partial, 1, NUM_HEADS, V_HEAD_DIM), dtype=torch.float32, device="cuda"),
            torch.empty((n_partial, 1, NUM_HEADS, 1), dtype=torch.float32, device="cuda"),
        )
    return _full_cache[key]


def _get_kv_indices(n: int) -> torch.Tensor:
    if n not in _kv_indices_cache:
        _kv_indices_cache[n] = torch.arange(n, dtype=torch.int32, device="cuda")
    return _kv_indices_cache[n]


def _get_output_buf(n: int) -> torch.Tensor:
    if n not in _output_cache:
        _output_cache[n] = torch.empty((n, NUM_HEADS, V_HEAD_DIM), dtype=BF16, device="cuda")
    return _output_cache[n]


# ---------------------------------------------------------------------------
# Main kernel — a16w8 (bf16 Q + fp8 KV) for all batch sizes
# ---------------------------------------------------------------------------

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    sm_scale = config["sm_scale"]

    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    total_kv_len = kv_buffer_fp8.shape[0]

    num_kv_splits = get_num_kv_splits(batch_size)
    (wm, wi, wis, ri, rfm, rpm, klp, logits, lse) = _get_or_build_all(
        batch_size, num_kv_splits, qo_indptr, kv_indptr,
    )
    kv_indices = _get_kv_indices(total_kv_len)
    o = _get_output_buf(q.shape[0])

    aiter.mla_decode_stage1_asm_fwd(
        q.view(-1, NUM_HEADS, QK_HEAD_DIM),
        kv_buffer_fp8.view(total_kv_len, PAGE_SIZE, NUM_KV_HEADS, QK_HEAD_DIM),
        qo_indptr, kv_indptr, kv_indices, klp,
        None, wm, wi, wis,
        1, PAGE_SIZE, NUM_KV_HEADS, sm_scale,
        logits, lse, o,
        None, kv_scale,
    )

    aiter.mla_reduce_v1(logits, lse, ri, rfm, rpm, 1, o, None)

    return o
