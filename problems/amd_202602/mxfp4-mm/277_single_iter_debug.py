#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#277: Single K-iteration HIP kernel that dumps A_fp4 and B_fp4 bytes + result.
Compare these bytes with what dynamic_mxfp4_quant produces.
If the bytes match but the result differs, the MFMA has a subtle issue.
If the bytes differ, we found the quant bug.
"""
import os, shutil, sys, json, torch
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.quant import dynamic_mxfp4_quant

_cfgs = {"N=2880-K=512": {"M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=4096-K=512": {"M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=7168-K=2048": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=3072-K=1536": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 3, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}}
try: _dev = arch_info.get_arch()
except: _dev = "gfx950"
_cd = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
os.makedirs(_cd, exist_ok=True)
for _sk, _cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json", "w") as f:
        json.dump(_cfg, f)

_MODULE = 'debug277'
_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bfloat16.h>
#include <torch/extension.h>
typedef int __attribute__((ext_vector_type(8))) vi8;
typedef float __attribute__((ext_vector_type(16))) vf16;
typedef hip_bfloat16 bf16;

// Single-iteration GEMM: takes PRE-QUANTIZED A_fp4 and B_fp4, does ONE MFMA
// Dumps A_fp4 bytes and B_fp4 bytes for lane 0 to debug buffers
extern "C" __global__ __launch_bounds__(64)
void debug277(
    const unsigned char* __restrict__ Aq,  // (M, Kh) pre-quantized FP4
    const unsigned char* __restrict__ As,  // (M, Ks) E8M0 scales
    const unsigned char* __restrict__ Bq,  // (N, Kh) pre-quantized FP4
    const unsigned char* __restrict__ Bs,  // (N, Ks) E8M0 scales
    float* __restrict__ out,               // (32, 32) output of ONE MFMA
    unsigned char* __restrict__ a_dump,    // (16,) A bytes for lane 0, grp 0
    unsigned char* __restrict__ b_dump,    // (16,) B bytes for lane 0, grp 0
    unsigned char* __restrict__ s_dump,    // (4,) a_scale0, a_scale1, b_scale0, b_scale1
    int M, int N, int K
) {
    int lane = threadIdx.x, l32 = lane % 32, grp = lane / 32;
    int Kh = K / 2, Ks = K / 32;

    // Load A FP4 for row l32, first 16 bytes (grp's half)
    union { vi8 v; unsigned char b[32]; } ab;
    for (int i = 0; i < 8; i++) ab.v[i] = 0;
    unsigned char asc = 127u;
    {
        int koff = grp * 16;  // first K-iteration
        if (l32 < M && koff < Kh) {
            for (int i = 0; i < 16 && koff+i < Kh; i++)
                ab.b[i] = Aq[(long)l32 * Kh + koff + i];
        }
        int ski = grp * 16 * 2 / 32;  // = grp
        if (l32 < M && ski < Ks)
            asc = As[(long)l32 * Ks + ski];
    }

    // Load B FP4 for col l32
    union { vi8 v; unsigned char b[32]; } bb;
    for (int i = 0; i < 8; i++) bb.v[i] = 0;
    unsigned char bsc = 127u;
    {
        int koff = grp * 16;
        if (l32 < N && koff < Kh) {
            for (int i = 0; i < 16 && koff+i < Kh; i++)
                bb.b[i] = Bq[(long)l32 * Kh + koff + i];
        }
        int ski = grp;
        if (l32 < N && ski < Ks)
            bsc = Bs[(long)l32 * Ks + ski];
    }

    // Dump lane 0 grp 0 bytes
    if (lane == 0) {
        for (int i = 0; i < 16; i++) a_dump[i] = ab.b[i];
        for (int i = 0; i < 16; i++) b_dump[i] = bb.b[i];
    }

    // Exchange scales
    unsigned int oasc = __shfl_xor((unsigned int)asc, 32);
    unsigned int obsc = __shfl_xor((unsigned int)bsc, 32);
    unsigned char as0 = grp == 0 ? asc : (unsigned char)oasc;
    unsigned char as1 = grp == 0 ? (unsigned char)oasc : asc;
    unsigned char bs0 = grp == 0 ? bsc : (unsigned char)obsc;
    unsigned char bs1 = grp == 0 ? (unsigned char)obsc : bsc;

    if (lane == 0) {
        s_dump[0] = as0; s_dump[1] = as1;
        s_dump[2] = bs0; s_dump[3] = bs1;
    }

    int asp = (int)as0 | ((int)as1 << 8);
    int bsp = (int)bs0 | ((int)bs1 << 8);

    vf16 acc = {}; for (int i = 0; i < 16; i++) acc[i] = 0.f;
    acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        ab.v, bb.v, acc, 4, 4, 0, asp, 0, bsp);

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            int r = grp * 4 + i * 8 + j;
            int c = l32;
            out[r * 32 + c] = acc[i * 4 + j];
        }
}

torch::Tensor run_debug(torch::Tensor Aq, torch::Tensor As, torch::Tensor Bq, torch::Tensor Bs, int M, int N, int K) {
    auto out = torch::zeros({32, 32}, torch::dtype(torch::kFloat32).device(Aq.device()));
    auto a_dump = torch::zeros({16}, torch::dtype(torch::kUInt8).device(Aq.device()));
    auto b_dump = torch::zeros({16}, torch::dtype(torch::kUInt8).device(Aq.device()));
    auto s_dump = torch::zeros({4}, torch::dtype(torch::kUInt8).device(Aq.device()));

    hipLaunchKernelGGL(debug277, dim3(1), dim3(64), 0, 0,
        (const unsigned char*)Aq.data_ptr(), (const unsigned char*)As.data_ptr(),
        (const unsigned char*)Bq.data_ptr(), (const unsigned char*)Bs.data_ptr(),
        (float*)out.data_ptr(), (unsigned char*)a_dump.data_ptr(),
        (unsigned char*)b_dump.data_ptr(), (unsigned char*)s_dump.data_ptr(),
        M, N, K);

    // Return all outputs packed
    auto result = torch::zeros({32*32 + 16 + 16 + 4}, torch::dtype(torch::kFloat32).device(Aq.device()));
    result[:32*32].copy_(out.flatten());
    // Pack byte dumps as floats
    auto ad = a_dump.cpu(); auto bd = b_dump.cpu(); auto sd = s_dump.cpu();
    // Return dumps separately via a dict? Just return out for now.
    return out;  // We'll print dumps from Python
}
"""

# Actually simpler: just use Python to compute the reference and compare
_CPP = r"""
#include <torch/extension.h>
// dummy
"""

# Don't need the HIP kernel for this - just compare the DATA
_bc = [None, None, None]
_first = [True]

for _m, _n, _k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
    try:
        _A = torch.randn((_m, _k), dtype=torch.bfloat16, device="cuda")
        _Bw = torch.zeros((_n//16, (_k//2)*16), dtype=torch.uint8, device="cuda")
        _Bws = torch.zeros((_n//32, _k), dtype=torch.uint8, device="cuda")
        gemm_a16wfp4_preshuffle(_A, _Bw, _Bws, prequant=True, dtype=torch.bfloat16)
    except: pass
torch.cuda.empty_cache()

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]
    dp = B_shuffle.data_ptr()
    if dp != _bc[0]:
        _bc[0] = dp
        _bc[1] = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        _bc[2] = B_scale_sh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)

    if _first[0] and k <= 512:
        _first[0] = False
        # Manual FP4 GEMM for element [0][0] using PyTorch
        # Compare with preshuffle result
        A_q, A_scale = dynamic_mxfp4_quant(A)
        A_q_u8 = A_q.view(torch.uint8)
        A_scale_u8 = A_scale.view(torch.uint8)[:m, :k//32].contiguous()
        B_q_u8 = B_q.view(torch.uint8)
        def _unshuffle(ssh, nn, kk):
            s = ssh.view(torch.uint8); sm, sn = s.shape
            return s.view(sm//32, sn//8, 4, 16, 2, 2).permute(0,5,3,1,4,2).contiguous().view(sm, sn)[:nn, :kk//32].contiguous()
        B_scale_u8 = _unshuffle(B_scale_sh, n, k)

        # Dequantize first 64 FP4 values of A row 0 and B row 0
        fp4_table = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        def deq_byte(b, scale):
            lo_code = b & 0xF
            hi_code = (b >> 4) & 0xF
            lo_sign = -1.0 if (lo_code & 8) else 1.0
            hi_sign = -1.0 if (hi_code & 8) else 1.0
            lo_val = fp4_table[lo_code & 7] * lo_sign * scale
            hi_val = fp4_table[hi_code & 7] * hi_sign * scale
            return lo_val, hi_val

        # Compute full dot product A_row[0] · B_row[0]
        Kh = k // 2
        Ks = k // 32
        dot = 0.0
        for kb in range(Kh):
            scale_idx = (kb * 2) // 32
            a_byte = A_q_u8[0, kb].item()
            b_byte = B_q_u8[0, kb].item()
            a_scale = 2.0 ** (A_scale_u8[0, scale_idx].item() - 127)
            b_scale = 2.0 ** (B_scale_u8[0, scale_idx].item() - 127)
            a_lo, a_hi = deq_byte(a_byte, a_scale)
            b_lo, b_hi = deq_byte(b_byte, b_scale)
            dot += a_lo * b_lo + a_hi * b_hi

        ps_result = gemm_a16wfp4_preshuffle(A, _bc[1], _bc[2], prequant=True, dtype=torch.bfloat16)
        print(f"[277] Manual dot A[0]·B[0] = {dot:.4f}", file=sys.stderr)
        print(f"[277] Preshuffle [0][0] = {ps_result[0,0].item():.4f}", file=sys.stderr)
        print(f"[277] A_scale[0,:4] = {A_scale_u8[0,:4].tolist()}", file=sys.stderr)
        print(f"[277] B_scale[0,:4] = {B_scale_u8[0,:4].tolist()}", file=sys.stderr)

    return gemm_a16wfp4_preshuffle(A, _bc[1], _bc[2], prequant=True, dtype=torch.bfloat16)
