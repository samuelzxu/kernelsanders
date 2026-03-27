#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4-MM #229: HIP MFMA with aiter quant + nibble-repacking B layout.
Tests whether B needs nibble-repacking even for (N,K/2) row-major format.
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
    "N=2880-K=512": {"M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}},
    "N=4096-K=512": {"M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}},
    "N=7168-K=2048": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}},
    "N=3072-K=1536": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 3, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}},
}
def _inject():
    try: dev = arch_info.get_arch()
    except: dev = "gfx950"
    cd = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
    os.makedirs(cd, exist_ok=True)
    for sk, cfg in _CONFIGS.items():
        with open(f"{cd}/{dev}-GEMM-A16WFP4_PRESHUFFLED-{sk}.json", "w") as f:
            json.dump(cfg, f)
try: _inject()
except: pass

_MODULE = 'mfma229'
_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bfloat16.h>
#include <torch/extension.h>

/* build 229: nibble-repacked B, pre-quantized A+B via aiter */
typedef int __attribute__((ext_vector_type(8))) vi8;
typedef float __attribute__((ext_vector_type(16))) vf16;
typedef hip_bfloat16 bf16;

// B needs nibble-repacking for MFMA transposed operand:
// lane32/2 = pair of N-rows, lane32%2 = nibble selector
// Each lane loads 16 bytes from B via LDS, repacking nibbles

extern "C" __global__ __launch_bounds__(64)
void kern229(
    const unsigned char* __restrict__ Aq,  // (M, K/2) packed FP4
    const unsigned char* __restrict__ As,  // (M, K/32) E8M0
    const unsigned char* __restrict__ Bq,  // (N, K/2) packed FP4
    const unsigned char* __restrict__ Bscl, // (N, K/32) E8M0
    bf16* __restrict__ C, float* __restrict__ Cs,
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

    // LDS for B tile repacking: 32 N-rows x 32 bytes
    __shared__ unsigned char lds_b[32 * 32];

    vf16 acc = {}; for (int i = 0; i < 16; i++) acc[i] = 0.f;

    for (int kp = ks; kp < ke; kp += 32) {
        // -- A: load pre-quantized FP4 directly --
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

        // -- B: load into LDS, then nibble-repack into registers --
        // Cooperative load: 64 threads load 32 rows x 32 bytes = 1024 bytes
        // Each thread loads 16 bytes
        {
            int thread_row = lane / 2;    // 0..31
            int thread_half = lane % 2;   // 0 or 1
            int gn = n0 + thread_row;
            int gk = kp + thread_half * 16;
            if (gn < N && gk < Kh) {
                const unsigned char* bp = Bq + gn * Kh + gk;
                int vld = Kh - gk; if (vld > 16) vld = 16;
                for (int i = 0; i < 16; i++)
                    lds_b[thread_row * 32 + thread_half * 16 + i] = (i < vld) ? bp[i] : 0;
            } else {
                for (int i = 0; i < 16; i++)
                    lds_b[thread_row * 32 + thread_half * 16 + i] = 0;
            }
        }
        __syncthreads();

        // Nibble-repack B from LDS
        union { vi8 v; unsigned char b[32]; } bb;
        for (int i = 0; i < 8; i++) bb.v[i] = 0;
        unsigned char bsc = 127u;
        {
            int npair = l32 / 2;     // which pair of N-rows [0..15]
            int nibsel = l32 % 2;    // which nibble
            int ln0 = npair * 2;     // first N-row in pair
            int ln1 = ln0 + 1;       // second N-row in pair

            for (int i = 0; i < 16; i++) {
                unsigned char byte0 = lds_b[ln0 * 32 + grp * 16 + i];
                unsigned char byte1 = lds_b[ln1 * 32 + grp * 16 + i];
                unsigned char nib0 = (nibsel == 0) ? (byte0 & 0xFu) : (byte0 >> 4);
                unsigned char nib1 = (nibsel == 0) ? (byte1 & 0xFu) : (byte1 >> 4);
                bb.b[i] = (nib0 & 0xFu) | ((nib1 & 0xFu) << 4);
            }

            // B scale: per the output column (which maps to N-row l32)
            int gn = n0 + l32;
            int ski = (kp + grp * 16) * 2 / 32;
            if (gn < N && ski < Ks)
                bsc = Bscl[gn * Ks + ski];
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

        __syncthreads();
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

extern "C" __global__ void kern229_red(const float* s, bf16* d, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) d[i] = bf16(s[i]);
}

torch::Tensor run229(torch::Tensor Aq, torch::Tensor As,
                     torch::Tensor Bq, torch::Tensor Bs, int ks) {
    int M = Aq.size(0), Kh = Aq.size(1), N = Bq.size(0), K = Kh * 2;
    auto C = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(Aq.device()));
    dim3 g((N+31)/32, (M+31)/32, ks);
    if (ks > 1) {
        auto Cs = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(Aq.device()));
        hipLaunchKernelGGL(kern229, g, dim3(64), 0, 0,
            (const unsigned char*)Aq.data_ptr(), (const unsigned char*)As.data_ptr(),
            (const unsigned char*)Bq.data_ptr(), (const unsigned char*)Bs.data_ptr(),
            (bf16*)C.data_ptr(), (float*)Cs.data_ptr(), M, N, K, ks);
        int tot = M * N;
        hipLaunchKernelGGL(kern229_red, dim3((tot+255)/256), dim3(256), 0, 0,
            (const float*)Cs.data_ptr(), (bf16*)C.data_ptr(), tot);
    } else {
        hipLaunchKernelGGL(kern229, g, dim3(64), 0, 0,
            (const unsigned char*)Aq.data_ptr(), (const unsigned char*)As.data_ptr(),
            (const unsigned char*)Bq.data_ptr(), (const unsigned char*)Bs.data_ptr(),
            (bf16*)C.data_ptr(), (float*)nullptr, M, N, K, 1);
    }
    return C;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor run229(torch::Tensor Aq, torch::Tensor As, torch::Tensor Bq, torch::Tensor Bs, int ks);
"""

# Clear cache
_cb = os.path.expanduser("~/.cache/torch_extensions")
for d in os.listdir(_cb) if os.path.isdir(_cb) else []:
    cd = os.path.join(_cb, d, _MODULE)
    if os.path.isdir(cd): shutil.rmtree(cd, ignore_errors=True)

_mod = None
try:
    _mod = load_inline(name=_MODULE, cpp_sources=_CPP_SRC, cuda_sources=_HIP_SRC,
                       functions=['run229'], verbose=True,
                       extra_cuda_cflags=['-O3', '-w', '-mcumode', '--offload-arch=gfx950'])
    print("[229] OK", file=sys.stderr)
except Exception as e:
    print(f"[229] FAIL: {e}", file=sys.stderr)

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
    return s.view(sm//32, sn//8, 4, 16, 2, 2).permute(0,5,3,1,4,2).contiguous().view(sm, sn)[:n, :k//32].contiguous()

_KS = {(4,2880,512):1,(16,2112,7168):4,(32,4096,512):1,(32,2880,512):1,(64,7168,2048):2,(256,3072,1536):3}

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]
    if _mod is not None:
        A_q, A_scale = dynamic_mxfp4_quant(A)
        aq = A_q.view(torch.uint8)
        asc = A_scale.view(torch.uint8)[:m, :k//32].contiguous()
        bq = B_q.view(torch.uint8)
        bs = _unshuffle(B_scale_sh, n, k)
        ks = _KS.get((m, n, k), 1)
        return _mod.run229(aq, asc, bq, bs, ks)
    return _ps_gemm(A, B_shuffle, B_scale_sh, m, k, n)
