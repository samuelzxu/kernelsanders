#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4-MM #228: HIP MFMA kernel with aiter's quant (separate quant + MFMA GEMM).
Uses aiter's dynamic_mxfp4_quant for A → then custom HIP MFMA for the FP4 GEMM.
This tests whether the MFMA computation itself is correct (isolating from quant).
Falls back to preshuffle if HIP fails.
"""
import os, shutil, sys
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
os.environ['HSA_XNACK'] = '0'

import json, torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.quant import dynamic_mxfp4_quant

_CONFIGS = {
    "N=2880-K=512": {
        "M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
    "N=4096-K=512": {
        "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
    "N=7168-K=2048": {
        "M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
    "N=3072-K=1536": {
        "M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 3, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
}

def _inject_configs():
    try:
        dev = arch_info.get_arch()
    except Exception:
        dev = "gfx950"
    config_dir = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
    os.makedirs(config_dir, exist_ok=True)
    for shape_key, config in _CONFIGS.items():
        fpath = f"{config_dir}/{dev}-GEMM-A16WFP4_PRESHUFFLED-{shape_key}.json"
        with open(fpath, "w") as f:
            json.dump(config, f)

try:
    _inject_configs()
except Exception:
    pass


# ── HIP kernel: takes PRE-QUANTIZED FP4 A and B ──
_MODULE = 'mfma228'
_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bfloat16.h>
#include <torch/extension.h>

/* build 228a: pre-quantized A + B, no inline quant */
typedef int __attribute__((ext_vector_type(8))) vi8;
typedef float __attribute__((ext_vector_type(16))) vf16;
typedef hip_bfloat16 bf16;

// 32x32 single-wavefront tile
// A_q: (M,K/2) uint8 packed FP4, A_s: (M,K/32) uint8 E8M0
// B_q: (N,K/2) uint8 packed FP4, B_s: (N,K/32) uint8 E8M0
// Both use same register layout: lane32=row, group=16-byte K-half
extern "C" __global__ __launch_bounds__(64)
void kern228(
    const unsigned char* __restrict__ Aq,
    const unsigned char* __restrict__ As,
    const unsigned char* __restrict__ Bq,
    const unsigned char* __restrict__ Bscl,
    bf16* __restrict__ C,
    float* __restrict__ Cs,
    int M, int N, int K, int ksplit
) {
    int bn = blockIdx.x, bm = blockIdx.y, bk = blockIdx.z;
    int lane = threadIdx.x, l32 = lane % 32, grp = lane / 32;
    int m0 = bm * 32, n0 = bn * 32;
    int Kh = K / 2, Ks = K / 32;

    int kps = ((Kh + ksplit - 1) / ksplit + 31) / 32 * 32;
    int ks = bk * kps, ke = ks + kps;
    if (ke > Kh) ke = Kh;
    if (ks >= Kh) return;

    vf16 acc = {}; for (int i = 0; i < 16; i++) acc[i] = 0.f;

    for (int kp = ks; kp < ke; kp += 32) {
        // Load A: pre-quantized FP4
        union { vi8 v; unsigned char b[32]; } ab;
        for (int i = 0; i < 8; i++) ab.v[i] = 0;
        unsigned char asc = 127u;
        {
            int r = m0 + l32, koff = kp + grp * 16;
            if (r < M && koff < Kh) {
                int vld = Kh - koff; if (vld > 16) vld = 16;
                const unsigned char* ap = Aq + r * Kh + koff;
                for (int i = 0; i < vld; i++) ab.b[i] = ap[i];
            }
            int ski = (kp + grp * 16) * 2 / 32;
            if (m0 + l32 < M && ski < Ks)
                asc = As[(m0 + l32) * Ks + ski];
        }

        // Load B: pre-quantized FP4
        union { vi8 v; unsigned char b[32]; } bb;
        for (int i = 0; i < 8; i++) bb.v[i] = 0;
        unsigned char bsc = 127u;
        {
            int nr = n0 + l32, koff = kp + grp * 16;
            if (nr < N && koff < Kh) {
                int vld = Kh - koff; if (vld > 16) vld = 16;
                const unsigned char* bp = Bq + nr * Kh + koff;
                for (int i = 0; i < vld; i++) bb.b[i] = bp[i];
            }
            int ski = (kp + grp * 16) * 2 / 32;
            if (n0 + l32 < N && ski < Ks)
                bsc = Bscl[(n0 + l32) * Ks + ski];
        }

        // Exchange scales
        unsigned int oasc = __shfl_xor((unsigned int)asc, 32);
        unsigned int obsc = __shfl_xor((unsigned int)bsc, 32);
        unsigned char as0 = grp == 0 ? asc : (unsigned char)oasc;
        unsigned char as1 = grp == 0 ? (unsigned char)oasc : asc;
        unsigned char bs0 = grp == 0 ? bsc : (unsigned char)obsc;
        unsigned char bs1 = grp == 0 ? (unsigned char)obsc : bsc;

        int asp = (int)as0 | ((int)as1 << 8);
        int bsp = (int)bs0 | ((int)bs1 << 8);

        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            ab.v, bb.v, acc, 4, 4, 0, asp, 0, bsp);
    }

    if (ksplit > 1 && Cs) {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                int r = m0 + grp*4 + i*8 + j, c = n0 + l32;
                if (r < M && c < N) atomicAdd(&Cs[r*N+c], acc[i*4+j]);
            }
    } else {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                int r = m0 + grp*4 + i*8 + j, c = n0 + l32;
                if (r < M && c < N) C[r*N+c] = bf16(acc[i*4+j]);
            }
    }
}

extern "C" __global__ void kern228_red(const float* s, bf16* d, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) d[i] = bf16(s[i]);
}

torch::Tensor run228(torch::Tensor Aq, torch::Tensor As,
                     torch::Tensor Bq, torch::Tensor Bs, int ks) {
    int M = Aq.size(0), Kh = Aq.size(1), N = Bq.size(0), K = Kh * 2;
    auto C = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(Aq.device()));
    dim3 g((N+31)/32, (M+31)/32, ks);
    if (ks > 1) {
        auto Cs = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(Aq.device()));
        hipLaunchKernelGGL(kern228, g, dim3(64), 0, 0,
            (const unsigned char*)Aq.data_ptr(), (const unsigned char*)As.data_ptr(),
            (const unsigned char*)Bq.data_ptr(), (const unsigned char*)Bs.data_ptr(),
            (bf16*)C.data_ptr(), (float*)Cs.data_ptr(), M, N, K, ks);
        int tot = M * N;
        hipLaunchKernelGGL(kern228_red, dim3((tot+255)/256), dim3(256), 0, 0,
            (const float*)Cs.data_ptr(), (bf16*)C.data_ptr(), tot);
    } else {
        hipLaunchKernelGGL(kern228, g, dim3(64), 0, 0,
            (const unsigned char*)Aq.data_ptr(), (const unsigned char*)As.data_ptr(),
            (const unsigned char*)Bq.data_ptr(), (const unsigned char*)Bs.data_ptr(),
            (bf16*)C.data_ptr(), (float*)nullptr, M, N, K, 1);
    }
    return C;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor run228(torch::Tensor Aq, torch::Tensor As,
                     torch::Tensor Bq, torch::Tensor Bs, int ks);
"""

# Clear cache and compile
_cache_base = os.path.expanduser("~/.cache/torch_extensions")
for d in os.listdir(_cache_base) if os.path.isdir(_cache_base) else []:
    cd = os.path.join(_cache_base, d, _MODULE)
    if os.path.isdir(cd):
        shutil.rmtree(cd, ignore_errors=True)

_mod = None
try:
    _mod = load_inline(
        name=_MODULE,
        cpp_sources=_CPP_SRC,
        cuda_sources=_HIP_SRC,
        functions=['run228'],
        verbose=True,
        extra_cuda_cflags=['-O3', '-w', '-mcumode', '--offload-arch=gfx950'],
    )
    print("[228] HIP kernel compiled OK", file=sys.stderr)
except Exception as e:
    print(f"[228] compile FAIL: {e}", file=sys.stderr)


# Preshuffle fallback
_bc = [None, None, None]

def _ps_gemm(A, Bsh, Bssh, m, k, n):
    key = (Bsh.data_ptr(), Bssh.data_ptr())
    if key != _bc[0]:
        _bc[0] = key
        _bc[1] = Bsh.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        _bc[2] = Bssh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
    return gemm_a16wfp4_preshuffle(A, _bc[1], _bc[2], prequant=True, dtype=torch.bfloat16)


def _unshuffle(ssh, n, k):
    s = ssh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm//32, sn//8, 4, 16, 2, 2).permute(0,5,3,1,4,2).contiguous().view(sm, sn)
    return s[:n, :k//32].contiguous()


_KSPLIT = {
    (4, 2880, 512): 1, (16, 2112, 7168): 4,
    (32, 4096, 512): 1, (32, 2880, 512): 1,
    (64, 7168, 2048): 2, (256, 3072, 1536): 3,
}


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _mod is not None:
        # Use aiter's quant for A (known correct)
        A_q, A_scale = dynamic_mxfp4_quant(A)
        A_q_uint8 = A_q.view(torch.uint8)
        A_scale_uint8 = A_scale.view(torch.uint8)[:m, :k//32].contiguous()

        # B: unshuffle from task format
        B_q_uint8 = B_q.view(torch.uint8)
        B_scale_uint8 = _unshuffle(B_scale_sh, n, k)

        ks = _KSPLIT.get((m, n, k), 1)
        return _mod.run228(A_q_uint8, A_scale_uint8, B_q_uint8, B_scale_uint8, ks)

    return _ps_gemm(A, B_shuffle, B_scale_sh, m, k, n)
