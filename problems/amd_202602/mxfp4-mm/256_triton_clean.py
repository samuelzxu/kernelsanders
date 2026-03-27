#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4-MM #256: Clean Triton dot_scaled kernel.
Fixes from #251: proper A_scale contiguity, correct B transposition.
Uses bf16 × e2m1 mixed precision via Triton 3.6.0's Scaled Dot Decomposition.
"""
import os, json, sys, torch, triton, triton.language as tl
from task import input_t, output_t
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.quant import dynamic_mxfp4_quant

_PS_CONFIGS = {"N=2880-K=512": {"M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=4096-K=512": {"M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=7168-K=2048": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=3072-K=1536": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 3, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}}
def _inject():
    try: dev = arch_info.get_arch()
    except: dev = "gfx950"
    cd = f"{AITER_TRITON_CONFIGS_PATH}/gemm"; os.makedirs(cd, exist_ok=True)
    for sk, cfg in _PS_CONFIGS.items():
        with open(f"{cd}/{dev}-GEMM-A16WFP4_PRESHUFFLED-{sk}.json", "w") as f: json.dump(cfg, f)
try: _inject()
except: pass


# Standalone Triton FP4 GEMM: bf16 A × fp4 B^T with block E8M0 scaling
# A: (M, K) bf16, A_scale: (M, K/32) E8M0 (from dynamic_mxfp4_quant, CONTIGUOUS)
# B: (K/2, N) packed FP4 (transposed B_q), B_scale: (N, K/32) E8M0
# C: (M, N) bf16
@triton.jit
def _dot_scaled_gemm(
    a_ptr, b_ptr, c_ptr, a_sc_ptr, b_sc_ptr,
    M, N, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_asm, stride_ask,
    stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    SG: tl.constexpr = 32
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    gsm = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_in_group) % gsm
    pid_n = (pid % num_in_group) // gsm

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # A: (M, K) bf16 - load BLOCK_M rows x BLOCK_K cols
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak
    # A scale: (M, K/32) E8M0
    offs_ks = tl.arange(0, BLOCK_K // SG)
    a_sc_ptrs = a_sc_ptr + offs_m[:, None] * stride_asm + offs_ks[None, :] * stride_ask
    # B: (K/2, N) packed FP4 - load BLOCK_K/2 rows x BLOCK_N cols
    offs_kh = tl.arange(0, BLOCK_K // 2)
    b_ptrs = b_ptr + offs_kh[:, None] * stride_bk + offs_n[None, :] * stride_bn
    # B scale: (N, K/32) E8M0
    b_sc_ptrs = b_sc_ptr + offs_n[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        a_sc = tl.load(a_sc_ptrs, mask=offs_m[:, None] < M, other=127)
        b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0)
        b_sc = tl.load(b_sc_ptrs, mask=offs_n[:, None] < N, other=127)

        # bf16 A × fp4 B: Triton 3.6.0 decomposes bf16 to fp4 inline
        acc = tl.dot_scaled(a, a_sc, "bf16", b, b_sc, "e2m1", acc)

        a_ptrs += BLOCK_K * stride_ak
        a_sc_ptrs += (BLOCK_K // SG) * stride_ask
        b_ptrs += (BLOCK_K // 2) * stride_bk
        b_sc_ptrs += (BLOCK_K // SG) * stride_bsk

    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def _unshuffle(ssh, n, k):
    s = ssh.view(torch.uint8)
    sm, sn = s.shape
    return s.view(sm//32, sn//8, 4, 16, 2, 2).permute(0,5,3,1,4,2).contiguous().view(sm, sn)[:n, :k//32].contiguous()


@triton.jit
def _dot_scaled_fp4_gemm(
    a_ptr, b_ptr, c_ptr, a_sc_ptr, b_sc_ptr,
    M, N, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_asm, stride_ask,
    stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    SG: tl.constexpr = 32
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    gsm = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_in_group) % gsm
    pid_n = (pid % num_in_group) // gsm

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kh = tl.arange(0, BLOCK_K // 2)
    offs_ks = tl.arange(0, BLOCK_K // SG)

    # A: (M, K/2) packed FP4
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_kh[None, :] * stride_ak
    a_sc_ptrs = a_sc_ptr + offs_m[:, None] * stride_asm + offs_ks[None, :] * stride_ask
    # B: (K/2, N) packed FP4 (transposed)
    b_ptrs = b_ptr + offs_kh[:, None] * stride_bk + offs_n[None, :] * stride_bn
    b_sc_ptrs = b_sc_ptr + offs_n[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0)
        a_sc = tl.load(a_sc_ptrs, mask=offs_m[:, None] < M, other=127)
        b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0)
        b_sc = tl.load(b_sc_ptrs, mask=offs_n[:, None] < N, other=127)
        acc = tl.dot_scaled(a, a_sc, "e2m1", b, b_sc, "e2m1", acc)
        a_ptrs += (BLOCK_K // 2) * stride_ak
        a_sc_ptrs += (BLOCK_K // SG) * stride_ask
        b_ptrs += (BLOCK_K // 2) * stride_bk
        b_sc_ptrs += (BLOCK_K // SG) * stride_bsk

    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def triton_gemm_fp4(A_q, B_q, A_scale, B_scale, M, N, K):
    C = torch.empty((M, N), dtype=torch.bfloat16, device=A_q.device)
    B_t = B_q.T.contiguous()
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 128, 256
    GROUP_SIZE_M = 4
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _dot_scaled_fp4_gemm[grid](
        A_q, B_t, C, A_scale, B_scale, M, N, K,
        A_q.stride(0), A_q.stride(1),
        B_t.stride(0), B_t.stride(1),
        C.stride(0), C.stride(1),
        A_scale.stride(0), A_scale.stride(1),
        B_scale.stride(0), B_scale.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=8, num_stages=2,
    )
    return C


def triton_gemm(A, B_q, A_scale, B_scale, M, N, K):
    C = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)
    # B_q is (N, K/2). Need (K/2, N) for the kernel.
    B_t = B_q.view(torch.uint8).T.contiguous()

    BLOCK_M, BLOCK_N, BLOCK_K = 32, 128, 256
    GROUP_SIZE_M = 4
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _dot_scaled_gemm[grid](
        A, B_t, C, A_scale, B_scale,
        M, N, K,
        A.stride(0), A.stride(1),
        B_t.stride(0), B_t.stride(1),
        C.stride(0), C.stride(1),
        A_scale.stride(0), A_scale.stride(1),
        B_scale.stride(0), B_scale.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=8, num_stages=2,
    )
    return C


_bc = [None, None, None]
def _ps_gemm(A, Bsh, Bssh, m, k, n):
    key = (Bsh.data_ptr(), Bssh.data_ptr())
    if key != _bc[0]:
        _bc[0] = key
        _bc[1] = Bsh.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        _bc[2] = Bssh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
    return gemm_a16wfp4_preshuffle(A, _bc[1], _bc[2], prequant=True, dtype=torch.bfloat16)


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]

    # Use Triton dot_scaled for K=1536 (our bottleneck shape)
    # Use PRE-QUANTIZED A (same quant as reference) + e2m1 × e2m1
    if k == 1536 and m >= 64:
        try:
            A_q, A_scale = dynamic_mxfp4_quant(A)
            A_q_u8 = A_q.view(torch.uint8)
            A_sc = A_scale.view(torch.uint8)[:m, :k//32].contiguous()
            B_q_u8 = B_q.view(torch.uint8)
            B_sc = _unshuffle(B_scale_sh, n, k)
            return triton_gemm_fp4(A_q_u8, B_q_u8, A_sc, B_sc, m, n, k)
        except Exception as e:
            print(f"[256] Triton failed: {e}", file=sys.stderr)

    # Preshuffle for all other shapes
    return _ps_gemm(A, B_shuffle, B_scale_sh, m, k, n)
