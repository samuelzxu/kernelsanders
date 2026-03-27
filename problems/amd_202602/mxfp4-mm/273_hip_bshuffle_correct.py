#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#273: HIP MFMA with CORRECT B_shuffle loading.
B_w = (N//16, K_packed*16). The shuffle format interleaves 16 N-rows per super-row.
Loading pattern derived from the preshuffle kernel's permute:
  B_w[super_row][kb*512 + kh*256 + n_within*16 + i]
  → MFMA lane l32 gets N-row (super_row*16 + n_within), K-byte (kb*32 + kh*16 + i)
For 32x32 tile with BSK=64 (one MFMA K-block):
  kb=0, kh=grp, so: B_w[sr][grp*256 + nw*16 + i]
"""
import os, shutil, sys, json, torch
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
os.environ['HSA_XNACK'] = '0'

from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle

_cfgs = {"N=2880-K=512": {"M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=4096-K=512": {"M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=7168-K=2048": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=3072-K=1536": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 3, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}}
try: _dev = arch_info.get_arch()
except: _dev = "gfx950"
_cd = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
os.makedirs(_cd, exist_ok=True)
for _sk, _cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json", "w") as f:
        json.dump(_cfg, f)

_MODULE = 'mfma273c'
_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bfloat16.h>
#include <torch/extension.h>

/* #273: Correct B_shuffle loading for MFMA */
typedef int __attribute__((ext_vector_type(8))) vi8;
typedef float __attribute__((ext_vector_type(16))) vf16;
typedef hip_bfloat16 bf16;

// Software FP4 quant (exact _mxfp4_quant_op)
__device__ __forceinline__ unsigned char sw_fp4(float val, float qscale) {
    float qx = val * qscale;
    unsigned int bits = __float_as_uint(qx);
    unsigned int s = bits & 0x80000000u;
    bits ^= s;
    float ax = __uint_as_float(bits);
    unsigned char r;
    if (ax >= 6.0f) { r = 0x7; }
    else if (ax < 1.0f) {
        float dx = ax + __uint_as_float(0x4A800000u);
        unsigned int db = __float_as_uint(dx) - 0x4A800000u;
        r = (unsigned char)(db & 0xFF);
    } else {
        unsigned int nx = bits;
        unsigned int mo = (nx >> 22) & 1;
        nx += (unsigned int)((int)((1-127)<<23) + (1<<21) - 1);
        nx += mo;
        nx >>= 22;
        r = (unsigned char)(nx & 0xFF);
    }
    r |= (unsigned char)(s >> 28);
    return r;
}

__device__ void quant32(const bf16* src, int valid, unsigned char* out16, unsigned char* e8m0) {
    float v[32]; float mx = 0.f;
    for (int i = 0; i < 32; i++) {
        v[i] = (i < valid) ? static_cast<float>(src[i]) : 0.f;
        mx = fmaxf(mx, fabsf(v[i]));
    }
    if (mx == 0.f) { *e8m0 = 0; for (int i=0;i<16;i++) out16[i]=0; return; }
    unsigned int ai = __float_as_uint(mx);
    ai = (ai + 0x200000u) & 0xFF800000u;
    int be = (int)((ai >> 23) & 0xFF);
    int e = be - 2; if (e<0) e=0; if (e>255) e=255;
    *e8m0 = (unsigned char)e;
    int ne = 129 - be + 127; if (ne<1) ne=1; if (ne>254) ne=254;
    float qs = __uint_as_float(((unsigned int)ne) << 23);
    for (int i = 0; i < 16; i++) {
        unsigned char lo = sw_fp4(v[2*i], qs);
        unsigned char hi = sw_fp4(v[2*i+1], qs);
        out16[i] = (lo & 0xF) | ((hi & 0xF) << 4);
    }
}

// 32x32 MFMA tile with CORRECT B_shuffle loading
// B_w: (N//16, K_packed*16) preshuffle format
// B_ws: (N//32, K) preshuffle scale format
extern "C" __global__ __launch_bounds__(64)
void kern273(
    const bf16* __restrict__ A,
    const unsigned char* __restrict__ Bw,   // (N//16, Kh*16) preshuffle
    const unsigned char* __restrict__ Bws,  // (N//32, K) preshuffle scale
    bf16* __restrict__ C, float* __restrict__ Cs,
    int M, int N, int K, int ksplit
) {
    int bn = blockIdx.x, bm = blockIdx.y, bk = blockIdx.z;
    int lane = threadIdx.x, l32 = lane % 32, grp = lane / 32;
    int m0 = bm * 32, n0 = bn * 32;
    int Kh = K / 2, Ks = K / 32;

    int kps = ((Kh + ksplit - 1) / ksplit + 31) / 32 * 32;
    int ks_start = bk * kps, ks_end = ks_start + kps;
    if (ks_end > Kh) ks_end = Kh;
    if (ks_start >= Kh) return;

    // B_w stride: each super-row has Kh * 16 bytes
    int bw_stride = Kh * 16;
    // B_ws stride: each row (32 N-rows) has K bytes
    int bws_stride = K;

    // Lane l32 maps to N-row (n0 + l32)
    int n_local = l32;
    int super_row = (n0 + n_local) / 16;
    int n_within = (n0 + n_local) % 16;

    // For B_ws: scale row = (n0 + n_local) / 32, n_within_32 = (n0 + n_local) % 32
    int scale_row = (n0 + n_local) / 32;
    int n_within_32 = (n0 + n_local) % 32;

    vf16 acc = {}; for (int i = 0; i < 16; i++) acc[i] = 0.f;

    for (int kp = ks_start; kp < ks_end; kp += 32) {
        // A: inline software FP4 quant
        union { vi8 v; unsigned char b[32]; } ab;
        for (int i = 0; i < 8; i++) ab.v[i] = 0;
        unsigned char asc = 0u;
        {
            int r = m0 + l32;
            int k_fp4 = (kp + grp * 16) * 2;
            if (r < M && k_fp4 < K) {
                int vld = K - k_fp4; if (vld > 32) vld = 32;
                quant32(A + (long)r * K + k_fp4, vld, ab.b, &asc);
            }
        }

        // B: load from B_shuffle (preshuffle format) with CORRECT indexing
        // From the Triton permute analysis:
        // B_w[super_row][kp*16 + grp*256 + n_within*16 + i]
        // Wait - need to account for kp offset within the super-row
        // The flat_k index for K-byte position (kp + grp*16 + i) within the super-row:
        // In B_w, the bytes for a super-row are laid out as:
        //   for each k_byte in [0, Kh):
        //     16 bytes for n_within [0..15]
        // So: B_w[sr][k_byte * 16 + n_within] = B_q[sr*16 + n_within][k_byte]
        // For our lane: k_byte = kp + grp*16 + i, n_within is fixed
        union { vi8 v; unsigned char b[32]; } bb;
        for (int i = 0; i < 8; i++) bb.v[i] = 0;
        unsigned char bsc = 0u;
        {
            int koff = kp + grp * 16;
            if (n0 + n_local < N && koff < Kh) {
                const unsigned char* sr_ptr = Bw + (long)super_row * bw_stride;
                int vld = Kh - koff; if (vld > 16) vld = 16;
                for (int i = 0; i < vld; i++) {
                    // B_w layout from shuffle_weight permute:
                    // B_w[sr][kb*512 + kh*256 + nw*16 + ki]
                    // where kb = (koff+i)/32, kh = ((koff+i)%32)/16, ki = (koff+i)%16
                    int k_byte = koff + i;
                    int kb = k_byte / 32;
                    int kh = (k_byte % 32) / 16;
                    int ki = k_byte % 16;
                    bb.b[i] = sr_ptr[kb * 512 + kh * 256 + n_within * 16 + ki];
                }
            }
            // B scale: now in UNSHUFFLED (N, K//32) row-major format
            int ski = (kp + grp * 16) * 2 / 32;
            if (n0 + n_local < N && ski < Ks) {
                bsc = Bws[(long)(n0 + n_local) * Ks + ski];
            }
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
                if (r < M && c < N) atomicAdd(&Cs[(long)r*N+c], acc[i*4+j]);
            }
    } else {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                int r = m0 + grp*4 + i*8 + j, c = n0 + l32;
                if (r < M && c < N) C[(long)r*N+c] = bf16(acc[i*4+j]);
            }
    }
}

extern "C" __global__ void kern273_red(const float* s, bf16* d, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) d[i] = bf16(s[i]);
}

torch::Tensor run273(torch::Tensor A, torch::Tensor Bw, torch::Tensor Bws, int ks) {
    int M = A.size(0), K = A.size(1);
    int N = Bw.size(0) * 16;
    auto C = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A.device()));
    dim3 g((N+31)/32, (M+31)/32, ks);
    if (ks > 1) {
        auto Cs = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(A.device()));
        hipLaunchKernelGGL(kern273, g, dim3(64), 0, 0,
            (const bf16*)A.data_ptr(), (const unsigned char*)Bw.data_ptr(),
            (const unsigned char*)Bws.data_ptr(), (bf16*)C.data_ptr(),
            (float*)Cs.data_ptr(), M, N, K, ks);
        int tot = M * N;
        hipLaunchKernelGGL(kern273_red, dim3((tot+255)/256), dim3(256), 0, 0,
            (const float*)Cs.data_ptr(), (bf16*)C.data_ptr(), tot);
    } else {
        hipLaunchKernelGGL(kern273, g, dim3(64), 0, 0,
            (const bf16*)A.data_ptr(), (const unsigned char*)Bw.data_ptr(),
            (const unsigned char*)Bws.data_ptr(), (bf16*)C.data_ptr(),
            (float*)nullptr, M, N, K, 1);
    }
    return C;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor run273(torch::Tensor A, torch::Tensor Bw, torch::Tensor Bws, int ks);
"""

_cb = os.path.expanduser("~/.cache/torch_extensions")
for d in os.listdir(_cb) if os.path.isdir(_cb) else []:
    cd = os.path.join(_cb, d, _MODULE)
    if os.path.isdir(cd): shutil.rmtree(cd, ignore_errors=True)

_mod = None
try:
    _mod = load_inline(name=_MODULE, cpp_sources=_CPP_SRC, cuda_sources=_HIP_SRC,
                       functions=['run273'], verbose=True,
                       extra_cuda_cflags=['-O3', '-w', '-mcumode', '--offload-arch=gfx950'])
    print("[273] OK", file=sys.stderr)
except Exception as e:
    print(f"[273] FAIL: {e}", file=sys.stderr)

# Preshuffle fallback
_bc = [None, None, None]
def _ps_gemm(A, Bsh, Bssh, m, k, n):
    key = (Bsh.data_ptr(), Bssh.data_ptr())
    if key != _bc[0]:
        _bc[0] = key
        _bc[1] = Bsh.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        _bc[2] = Bssh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
    return gemm_a16wfp4_preshuffle(A, _bc[1], _bc[2], prequant=True, dtype=torch.bfloat16)

_KS = {(4,2880,512):1,(16,2112,7168):4,(32,4096,512):1,(32,2880,512):1,(64,7168,2048):2,(256,3072,1536):3}

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]

    if _mod is not None:
        B_w = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        # Use UNSHUFFLED B_scale (standard (N, K//32) format) for the HIP kernel
        # since we don't replicate the in-kernel scale permute
        def _unshuffle_scale(ssh, nn, kk):
            s = ssh.view(torch.uint8)
            sm, sn = s.shape
            return s.view(sm//32, sn//8, 4, 16, 2, 2).permute(0,5,3,1,4,2).contiguous().view(sm, sn)[:nn, :kk//32].contiguous()
        B_scale_flat = _unshuffle_scale(B_scale_sh, n, k)
        ks = _KS.get((m, n, k), 1)
        return _mod.run273(A, B_w, B_scale_flat, ks)

    return _ps_gemm(A, B_shuffle, B_scale_sh, m, k, n)
