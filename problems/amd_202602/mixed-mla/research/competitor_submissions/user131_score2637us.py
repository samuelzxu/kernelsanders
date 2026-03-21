import math
import time
import random

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from task import input_t, output_t
from reference import KVCache, Config
from utils import verbose_allclose


# def set_seed(seed=0):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)


qk_rope_head_dim = 64

# start = time.perf_counter_ns()

theta = 10000 ** (-torch.arange(0, qk_rope_head_dim // 2, dtype=torch.bfloat16, device="cuda") / (qk_rope_head_dim // 2))
seq_idx = torch.arange(0, 8192, device="cuda")
idx_theta = torch.einsum('s,d->sd', seq_idx, theta)
idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)
pre_cos = idx_theta2.cos().to(torch.bfloat16).unsqueeze(1)
pre_sin = idx_theta2.sin().to(torch.bfloat16).unsqueeze(1)

# end = time.perf_counter_ns()
# print(end - start)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope(x: torch.Tensor, theta: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    seq_len = x.size(-2)
    seq_idx = torch.arange(start_pos, start_pos + seq_len, device=x.device)
    idx_theta = torch.einsum('s,d->sd', seq_idx, theta)
    idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)
    cos = idx_theta2.cos().to(torch.bfloat16)
    sin = idx_theta2.sin().to(torch.bfloat16)
    return x * cos + rotate_half(x) * sin


def apply_rotary_emb(x: torch.Tensor, theta: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    seq_len = x.size(-3)
    seq_idx = torch.arange(start_pos, start_pos + seq_len, device=x.device)
    idx_theta = torch.einsum('s,d->sd', seq_idx, theta)
    idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)
    cos = idx_theta2.cos().to(torch.bfloat16).unsqueeze(1)
    sin = idx_theta2.sin().to(torch.bfloat16).unsqueeze(1)
    return x * cos + rotate_half(x) * sin


def apply_rotary_emb_pre(x: torch.Tensor, theta: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    seq_len = x.size(-3)
    cos = pre_cos[start_pos:start_pos + seq_len]
    sin = pre_sin[start_pos:start_pos + seq_len]
    return x * cos + rotate_half(x) * sin


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x):
    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_stages = 2

    y = torch.empty_like(x)

    softmax_kernel[(n_rows,)](
        y,
        x,
        x.stride(-2),
        y.stride(-2),
        n_rows,
        n_cols,
        BLOCK_SIZE,
        num_stages)
    return y


def mla_absorb(x: torch.Tensor, kv_cache: KVCache, config: Config) -> torch.Tensor:
    bsz, seqlen, _ = x.size()

    ################################################################################################
    # get q_nope, q_rope
    ################################################################################################
    # q = self.wq_b(self.q_norm(self.wq_a(x)))
    q = F.linear(F.linear(x, config.Q_proj_down_weight), config.Q_proj_up_weight)
    q = q.view(bsz, seqlen, config.n_heads, config.qk_nope_head_dim + config.qk_rope_head_dim)
    q_nope, q_rope = torch.split(q, [config.qk_nope_head_dim, config.qk_rope_head_dim], dim=-1)

    ################################################################################################
    # get kv, k_rope
    ################################################################################################
    # kv = self.wkv_a(x)
    kv = F.linear(x, config.KV_proj_down_weight)
    kv, kv_len = kv_cache(kv)
    kv, k_rope = torch.split(kv, [config.kv_lora_rank, config.qk_rope_head_dim], dim=-1)

    ################################################################################################
    # apply rope
    ################################################################################################
    q_rope = apply_rotary_emb(q_rope, theta, start_pos=kv_len - 1)
    k_rope = apply_rotary_emb(k_rope.unsqueeze(2), theta)

    ################################################################################################
    # multiply q_nope with w_uk
    ################################################################################################
    wkv_b = config.KV_proj_up_weight.view(config.n_heads, -1, config.kv_lora_rank)
    # q_nope shape [128, 1, 128, 128], wkv_b shape [128, 128, 512], res shape [128, 1, 128, 512], flops 1.07 G
    q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :config.qk_nope_head_dim])

    ################################################################################################
    # attention
    ################################################################################################
    # q_nope shape [128, 1, 128, 512], kv shape [128, 513, 512], res shape [128, 1, 128, 513], flops 4.3 G
    # q_nope shape [128, 1, 128, 64], kv shape [128, 513, 64], res shape [128, 1, 128, 513], flops 0.54 G
    scores = (torch.einsum("bshc,btc->bsht", q_nope, kv) +
              torch.einsum("bshr,btr->bsht", q_rope, k_rope.squeeze(2))) / math.sqrt(config.qk_rope_head_dim + config.qk_nope_head_dim)
    scores = softmax(scores)
    # scores shape [128, 1, 128, 513], kv shape [128, 513, 512], res shape [128, 1, 128, 512], flops 4.3 G
    x = torch.einsum("bsht,btc->bshc", scores, kv)

    # x shape [128, 1, 128, 512], wkv_b shape [128, 128, 512], res shape [128, 1, 128, 512], flops 1.07 G
    x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -config.v_head_dim:])

    ################################################################################################
    # output projection
    ################################################################################################
    x = F.linear(x.flatten(2), config.wo_weight)

    return x, kv_cache.get_data()


def custom_kernel(data: input_t) -> output_t:
    config, x, kv_cache = data
    return mla_absorb(x, kv_cache, config)


# if __name__ == '__main__':
#     set_seed()

#     from reference import generate_input, ref_kernel
#     from eval import copy_kv_cache

#     batchsize = 128
#     dim = 7168 
#     dq = 1536
#     prefill = 512
#     seed = 97

#     config, x, kv_cache = generate_input(batchsize, dim, dq, prefill, seed)
#     ref_kv_cache = copy_kv_cache(kv_cache, config.kv_cache_shape)

#     start = time.perf_counter_ns()
#     output, kv = mla_absorb(x, kv_cache, config)
#     end = time.perf_counter_ns()
#     print(end - start)
#     # print(output.shape)
#     # print(kv.shape)

#     start = time.perf_counter_ns()
#     ref_output, ref_kv = ref_kernel((config, x, ref_kv_cache))
#     end = time.perf_counter_ns()
#     print(end - start)
#     # print(ref_output.shape)
#     # print(ref_kv.shape)

#     # start = time.perf_counter_ns()
#     # ref_output, ref_kv = mla_navie(x, ref_kv_cache, config)
#     # end = time.perf_counter_ns()
#     # print(end - start)
#     # print(ref_output.shape)
#     # print(ref_kv.shape)

#     print(verbose_allclose(output, ref_output, rtol=2e-02, atol=8e-03))
#     print(verbose_allclose(kv, ref_kv, rtol=2e-02, atol=8e-03))
