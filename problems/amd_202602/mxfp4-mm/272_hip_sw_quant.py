#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#272: HIP MFMA kernel with SOFTWARE FP4 quant matching _mxfp4_quant_op EXACTLY.
Translates the complete Triton _mxfp4_quant_op bit manipulation to HIP C++.
Combined with confirmed MFMA register layout and scale packing.
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

_MODULE = 'mfma272e'
_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bfloat16.h>
#include <torch/extension.h>

/* #272: SOFTWARE FP4 quant matching _mxfp4_quant_op exactly */
typedef int __attribute__((ext_vector_type(8))) vi8;
typedef float __attribute__((ext_vector_type(16))) vf16;
typedef hip_bfloat16 bf16;

#define EXP_BIAS_FP32 127
#define EXP_BIAS_FP4 1
#define EBITS_F32 8
#define EBITS_FP4 2
#define MBITS_F32 23
#define MBITS_FP4 1

// Exact translation of _mxfp4_quant_op from aiter Triton source
// Converts a single float32 value to FP4 E2M1 (4-bit) given a pre-computed scale
__device__ __forceinline__ unsigned char sw_f32_to_fp4(float val, float quant_scale) {
    float qx = val * quant_scale;
    unsigned int qx_bits = __float_as_uint(qx);

    // Extract sign
    unsigned int s = qx_bits & 0x80000000u;
    qx_bits ^= s;  // make positive

    float qx_abs = __uint_as_float(qx_bits);

    unsigned char result;

    if (qx_abs >= 6.0f) {
        // Saturate
        result = 0x7;  // max FP4 = 6.0
    } else if (qx_abs < 1.0f) {
        // Denormal path: add magic number to round, then extract bits
        // denorm_exp = (127 - 1) + (23 - 1) + 1 = 149
        // denorm_mask_int = 149 << 23 = 0x4A800000
        float denormal_x = qx_abs + __uint_as_float(0x4A800000u);
        unsigned int den_bits = __float_as_uint(denormal_x);
        den_bits -= 0x4A800000u;
        result = (unsigned char)(den_bits & 0xFF);
    } else {
        // Normal path: round-to-nearest-even via bit manipulation
        unsigned int normal_x = qx_bits;
        unsigned int mant_odd = (normal_x >> (MBITS_F32 - MBITS_FP4)) & 1;
        // val_to_add = ((1 - 127) << 23) + (1 << 21) - 1 = 0xC0200000 - 1 + 0x200000
        // = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32) + (1 << 21) - 1
        // = (-126 << 23) + 2097151 = 0xC1000000 + 0x1FFFFF
        // Actually: (1-127) = -126. -126 << 23 as unsigned:
        // In the Triton code: val_to_add = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32) + (1 << 21) - 1
        // = ((1 - 127) << 23) + (1 << 21) - 1
        // = (-126 << 23) + 2097151
        // In unsigned: -126 << 23 = 0xFFFFFF82 << 23 ... this is tricky with signed shift
        // Let's compute: (1 - 127) = -126. As int32: -126. Shifted left 23: -126 * 8388608 = -1056964608
        // In hex: 0xC1000000. Plus (1<<21)-1 = 2097151 = 0x1FFFFF
        // Total: 0xC1000000 + 0x001FFFFF = 0xC11FFFFF
        int val_to_add = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32) + (1 << 21) - 1;
        normal_x = (unsigned int)((int)normal_x + val_to_add);
        normal_x += mant_odd;  // round-to-nearest-even
        normal_x >>= (MBITS_F32 - MBITS_FP4);
        result = (unsigned char)(normal_x & 0xFF);
    }

    // Add sign back
    unsigned char sign_fp4 = (unsigned char)(s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4));
    result |= sign_fp4;
    return result;
}

// Quantize 32 bf16 values to 16 packed FP4 bytes using SOFTWARE conversion
__device__ void sw_quant_32(const bf16* src, int valid, unsigned char* out16, unsigned char* out_scale) {
    float vals[32];
    float max_abs = 0.f;
    for (int i = 0; i < 32; i++) {
        vals[i] = (i < valid) ? static_cast<float>(src[i]) : 0.f;
        float av = fabsf(vals[i]);
        if (av > max_abs) max_abs = av;
    }

    // E8M0 scale (exact aiter formula)
    unsigned char e8m0;
    float quant_scale;
    if (max_abs == 0.f) {
        e8m0 = 0u;
        quant_scale = 1.0f;  // doesn't matter, all zeros
    } else {
        unsigned int ai = __float_as_uint(max_abs);
        ai = (ai + 0x200000u) & 0xFF800000u;
        int biased_exp = (int)((ai >> 23) & 0xFF);
        float scale_unbiased_f = __uint_as_float(((unsigned int)(biased_exp - 2)) << 23);
        // quant_scale = 2^(-scale_unbiased) = 1/2^(biased_exp - 2 - 127)
        // = 2^(129 - biased_exp)
        int neg_exp = 129 - biased_exp + 127;  // biased form of -(biased_exp-129)
        if (neg_exp < 1) neg_exp = 1;
        if (neg_exp > 254) neg_exp = 254;
        quant_scale = __uint_as_float(((unsigned int)neg_exp) << 23);
        int e8m0_val = biased_exp - 2;
        if (e8m0_val < 0) e8m0_val = 0;
        if (e8m0_val > 255) e8m0_val = 255;
        e8m0 = (unsigned char)e8m0_val;
    }
    *out_scale = e8m0;

    // Convert each pair to packed FP4 (evens in low nibble, odds in high nibble)
    for (int i = 0; i < 16; i++) {
        unsigned char lo = sw_f32_to_fp4(vals[2*i], quant_scale);
        unsigned char hi = sw_f32_to_fp4(vals[2*i+1], quant_scale);
        // TEST: swap nibble order to match potential MFMA convention
        out16[i] = ((lo & 0xF) << 4) | (hi & 0xF);
    }
}

// 32x32 single-wavefront MFMA tile
extern "C" __global__ __launch_bounds__(64)
void kern272(
    const bf16* __restrict__ A,
    const unsigned char* __restrict__ Bq,
    const unsigned char* __restrict__ Bs,
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

    vf16 acc = {}; for (int i = 0; i < 16; i++) acc[i] = 0.f;

    for (int kp = ks_start; kp < ks_end; kp += 32) {
        // A: SOFTWARE FP4 quant - data at bytes [0..15] for BOTH groups
        union { vi8 v; unsigned char b[32]; } ab;
        for (int i = 0; i < 32; i++) ab.b[i] = 0;
        unsigned char asc = 0u;
        {
            int r = m0 + l32;
            int k_fp4 = (kp + grp * 16) * 2;
            if (r < M && k_fp4 < K) {
                int valid = K - k_fp4; if (valid > 32) valid = 32;
                sw_quant_32(A + (long)r * K + k_fp4, valid, ab.b, &asc);
            }
        }

        // B: load pre-quantized FP4 - data at bytes [0..15] for BOTH groups
        union { vi8 v; unsigned char b[32]; } bb;
        for (int i = 0; i < 32; i++) bb.b[i] = 0;
        unsigned char bsc = 0u;
        {
            int nr = n0 + l32, koff = kp + grp * 16;
            if (nr < N && koff < Kh) {
                const unsigned char* bp = Bq + (long)nr * Kh + koff;
                int vld = Kh - koff; if (vld > 16) vld = 16;
                for (int i = 0; i < vld; i++) {
                    // TEST: swap nibbles in B bytes
                    unsigned char orig = bp[i];
                    bb.b[i] = ((orig & 0xF) << 4) | ((orig >> 4) & 0xF);
                }
            }
            int ski = (kp + grp * 16) * 2 / 32;
            if (n0 + l32 < N && ski < Ks)
                bsc = Bs[(long)(n0 + l32) * Ks + ski];
        }

        // Scale packing: each lane has its own scale at the correct byte position
        // grp=0 scale → byte[0], grp=1 scale → byte[1]
        // Exchange so each lane has BOTH scales
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

extern "C" __global__ void kern272_red(const float* s, bf16* d, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) d[i] = bf16(s[i]);
}

torch::Tensor run272(torch::Tensor A, torch::Tensor Bq, torch::Tensor Bs, int ks) {
    int M = A.size(0), K = A.size(1), N = Bq.size(0);
    auto C = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A.device()));
    dim3 g((N+31)/32, (M+31)/32, ks);
    if (ks > 1) {
        auto Cs = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(A.device()));
        hipLaunchKernelGGL(kern272, g, dim3(64), 0, 0,
            (const bf16*)A.data_ptr(), (const unsigned char*)Bq.data_ptr(),
            (const unsigned char*)Bs.data_ptr(), (bf16*)C.data_ptr(),
            (float*)Cs.data_ptr(), M, N, K, ks);
        int tot = M * N;
        hipLaunchKernelGGL(kern272_red, dim3((tot+255)/256), dim3(256), 0, 0,
            (const float*)Cs.data_ptr(), (bf16*)C.data_ptr(), tot);
    } else {
        hipLaunchKernelGGL(kern272, g, dim3(64), 0, 0,
            (const bf16*)A.data_ptr(), (const unsigned char*)Bq.data_ptr(),
            (const unsigned char*)Bs.data_ptr(), (bf16*)C.data_ptr(),
            (float*)nullptr, M, N, K, 1);
    }
    return C;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor run272(torch::Tensor A, torch::Tensor Bq, torch::Tensor Bs, int ks);
"""

_cb = os.path.expanduser("~/.cache/torch_extensions")
for d in os.listdir(_cb) if os.path.isdir(_cb) else []:
    cd = os.path.join(_cb, d, _MODULE)
    if os.path.isdir(cd): shutil.rmtree(cd, ignore_errors=True)

_mod = None
try:
    _mod = load_inline(name=_MODULE, cpp_sources=_CPP_SRC, cuda_sources=_HIP_SRC,
                       functions=['run272'], verbose=True,
                       extra_cuda_cflags=['-O3', '-w', '-mcumode', '--offload-arch=gfx950'])
    print("[272] OK", file=sys.stderr)
except Exception as e:
    print(f"[272] FAIL: {e}", file=sys.stderr)

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
    return s.view(sm//32, sn//8, 4, 16, 2, 2).permute(0,5,3,1,4,2).contiguous().view(sm, sn)[:n, :k//32].contiguous()

_KS = {(4,2880,512):1,(16,2112,7168):4,(32,4096,512):1,(32,2880,512):1,(64,7168,2048):2,(256,3072,1536):3}

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]

    if _mod is not None:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        # Use FRESH quant for B (not task's B_q which might differ)
        B_q_fresh, B_scale_fresh = dynamic_mxfp4_quant(B)
        bq = B_q_fresh.view(torch.uint8)
        bs = B_scale_fresh.view(torch.uint8)[:n, :k//32].contiguous()
        ks = _KS.get((m, n, k), 1)
        return _mod.run272(A, bq, bs, ks)

    return _ps_gemm(A, B_shuffle, B_scale_sh, m, k, n)
