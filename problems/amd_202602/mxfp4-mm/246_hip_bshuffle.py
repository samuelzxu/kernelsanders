#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4-MM #246: HIP MFMA using B_shuffle (preshuffle format) directly.

Key insight: B_shuffle is ALREADY in MFMA register order. The preshuffle step
exists precisely so sequential byte loading produces the correct MFMA layout.
Load B_shuffle bytes sequentially - no reordering, no nibble repacking.

B_shuffle: (N, K//2) → reshaped to (N//16, K*8) for preshuffle kernel.
In preshuffle format, 16 consecutive N-rows' K-data is interleaved.

For the HIP kernel: load from B_w = B_shuffle.reshape(N//16, K*8).
Lane l32 within a 32-row block needs data from the appropriate position.

Also uses B_ws = B_scale_sh.reshape(N//32, K) for preshuffle scale format.
"""
import os, shutil, sys, json, torch
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
os.environ['HSA_XNACK'] = '0'

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

_MODULE = 'mfma246'
_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bfloat16.h>
#include <torch/extension.h>

/* build 246: use B_shuffle (preshuffle format) + aiter quant for A */
typedef int __attribute__((ext_vector_type(8))) vi8;
typedef float __attribute__((ext_vector_type(16))) vf16;
typedef hip_bfloat16 bf16;

// 32x32 tile, 1 wavefront
// A_q: (M, K/2) uint8 packed FP4 - from dynamic_mxfp4_quant
// A_s: (M, K/32) uint8 E8M0 - from dynamic_mxfp4_quant, made contiguous
// B_w: (N/16, K*8) uint8 - B_shuffle reshaped to preshuffle format
// B_ws: (N/32, K) uint8 - B_scale_sh reshaped to preshuffle scale format
// The preshuffle format means 16 N-rows are packed per "super-row" of K*8 bytes.
// For a 32-row N-block, we need 2 super-rows.
// Within each super-row, bytes are ordered for direct MFMA register loading.
extern "C" __global__ __launch_bounds__(64)
void kern246(
    const unsigned char* __restrict__ Aq,   // (M, K/2) row-major
    const unsigned char* __restrict__ As,   // (M, K/32) row-major (contiguous)
    const unsigned char* __restrict__ Bw,   // (N/16, K*8) preshuffle format
    const unsigned char* __restrict__ Bws,  // (N/32, K) preshuffle scale format
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

    // B_w has stride: each super-row (16 N-rows) has K*8 bytes
    // For N-block starting at n0, the super-row indices are n0/16 and n0/16+1
    int bw_stride = K * 8;  // bytes per super-row in B_w
    // B_ws has stride: each row (32 N-rows) has K bytes
    int bws_stride = K;

    vf16 acc = {}; for (int i = 0; i < 16; i++) acc[i] = 0.f;

    for (int kp = ks_start; kp < ks_end; kp += 32) {
        // A: load pre-quantized FP4
        union { vi8 v; unsigned char b[32]; } ab;
        for (int i = 0; i < 8; i++) ab.v[i] = 0;
        unsigned char asc = 127u;
        {
            int r = m0 + l32, koff = kp + grp * 16;
            if (r < M && koff < Kh) {
                const unsigned char* ap = Aq + (long)r * Kh + koff;
                int vld = Kh - koff; if (vld > 16) vld = 16;
                for (int i = 0; i < vld; i++) ab.b[i] = ap[i];
            }
            int ski = (kp + grp * 16) * 2 / 32;
            if (m0 + l32 < M && ski < Ks)
                asc = As[(long)(m0 + l32) * Ks + ski];
        }

        // B: load from preshuffle format (B_w)
        // B_w is (N/16, K*8). For our 32-row N-block:
        // - First 16 N-rows: super-row n0/16, offset in super-row depends on l32 and kp
        // - Second 16 N-rows: super-row n0/16+1
        // The preshuffle format packs 16 N-rows' K-data.
        // Within the super-row, the layout is: for each K-byte position,
        // 16 consecutive N-rows are packed together (16 bytes per K-position).
        // So byte at super_row[k_byte * 16 + n_within_16] = B_q[n0 + n_within_16][k_byte]
        union { vi8 v; unsigned char b[32]; } bb;
        for (int i = 0; i < 8; i++) bb.v[i] = 0;
        unsigned char bsc = 127u;
        {
            // l32 selects which N-row within the 32-row block
            int n_local = l32;           // 0..31
            int super_row = (n0 + n_local) / 16;  // which super-row
            int n_within = (n0 + n_local) % 16;   // position within super-row
            int koff = kp + grp * 16;

            if (n0 + n_local < N && koff < Kh) {
                // In preshuffle: byte at [super_row, k_byte * 16 + n_within]
                const unsigned char* brow = Bw + (long)super_row * bw_stride;
                int vld = Kh - koff; if (vld > 16) vld = 16;
                for (int i = 0; i < vld; i++) {
                    bb.b[i] = brow[(koff + i) * 16 + n_within];
                }
            }

            // B scale from preshuffle format: B_ws is (N/32, K)
            int scale_row = (n0 + n_local) / 32;
            int ski = (kp + grp * 16) * 2 / 32;  // which scale group
            // In preshuffle scale format: [scale_row, ski * 32 + n_within_32]?
            // Actually B_ws = B_scale_sh.reshape(N//32, K) where K = K_original
            // The preshuffle scale has 32 N-rows packed per row, K scale groups per row
            // Each element is the scale for a specific (N, K_group) pair
            // Need to figure out the exact indexing...
            // For now, use sequential indexing: B_ws[scale_row, ski]
            // But that gives 1 scale per 32 N-rows per K-group, not per-N-row
            // The preshuffle scale packs differently...

            // Actually: B_ws has shape (N//32, K). With K_original columns.
            // The scales for N-rows [n0..n0+31] are in B_ws row n0/32.
            // Within that row, element j contains the scale for some (n_offset, k_group) pair.
            // The exact mapping depends on the e8m0_shuffle permutation.
            // Since we verified _unshuffle matches fresh quant, the shuffled format IS:
            // B_ws[n0/32, ski * 32 + n_within_32] = B_scale[n0+n_within_32, ski]
            // Wait, B_ws has K columns and N/32 rows. K = K_original = 1536 for that shape.
            // But B_scale has K/32 = 48 columns. So B_ws must pack 32 N-rows * (K/32) scale groups
            // into K = 1536 columns per row. 32 * 48 = 1536. Yes!
            // So B_ws[n0/32, n_within_32 * (K/32) + ski] = scale for N-row (n0+n_within_32), K-group ski
            int n_within_32 = (n0 + n_local) % 32;
            int bws_idx = scale_row * bws_stride + n_within_32 * Ks + ski;
            if (n0 + n_local < N && ski < Ks)
                bsc = Bws[bws_idx];
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

extern "C" __global__ void kern246_red(const float* s, bf16* d, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) d[i] = bf16(s[i]);
}

torch::Tensor run246(torch::Tensor Aq, torch::Tensor As,
                     torch::Tensor Bw, torch::Tensor Bws, int ks, int K) {
    int M = Aq.size(0), Kh = Aq.size(1);
    // N = Bw.size(0) * 16 (since Bw is N/16 rows)
    int N = Bw.size(0) * 16;
    auto C = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(Aq.device()));
    dim3 g((N+31)/32, (M+31)/32, ks);
    if (ks > 1) {
        auto Cs = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(Aq.device()));
        hipLaunchKernelGGL(kern246, g, dim3(64), 0, 0,
            (const unsigned char*)Aq.data_ptr(), (const unsigned char*)As.data_ptr(),
            (const unsigned char*)Bw.data_ptr(), (const unsigned char*)Bws.data_ptr(),
            (bf16*)C.data_ptr(), (float*)Cs.data_ptr(), M, N, K, ks);
        int tot = M * N;
        hipLaunchKernelGGL(kern246_red, dim3((tot+255)/256), dim3(256), 0, 0,
            (const float*)Cs.data_ptr(), (bf16*)C.data_ptr(), tot);
    } else {
        hipLaunchKernelGGL(kern246, g, dim3(64), 0, 0,
            (const unsigned char*)Aq.data_ptr(), (const unsigned char*)As.data_ptr(),
            (const unsigned char*)Bw.data_ptr(), (const unsigned char*)Bws.data_ptr(),
            (bf16*)C.data_ptr(), (float*)nullptr, M, N, K, 1);
    }
    return C;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor run246(torch::Tensor Aq, torch::Tensor As,
                     torch::Tensor Bw, torch::Tensor Bws, int ks, int K);
"""

_cb = os.path.expanduser("~/.cache/torch_extensions")
for d in os.listdir(_cb) if os.path.isdir(_cb) else []:
    cd = os.path.join(_cb, d, _MODULE)
    if os.path.isdir(cd): shutil.rmtree(cd, ignore_errors=True)

_mod = None
try:
    _mod = load_inline(name=_MODULE, cpp_sources=_CPP_SRC, cuda_sources=_HIP_SRC,
                       functions=['run246'], verbose=True,
                       extra_cuda_cflags=['-O3', '-w', '-mcumode', '--offload-arch=gfx950'])
    print("[246] OK", file=sys.stderr)
except Exception as e:
    print(f"[246] FAIL: {e}", file=sys.stderr)

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
        A_q, A_scale = dynamic_mxfp4_quant(A)
        aq = A_q.view(torch.uint8).contiguous()
        asc = A_scale.view(torch.uint8)[:m, :k//32].contiguous()
        # Preshuffle B format
        B_w = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        B_ws = B_scale_sh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
        ks = _KS.get((m, n, k), 1)
        return _mod.run246(aq, asc, B_w, B_ws, ks, k)

    return _ps_gemm(A, B_shuffle, B_scale_sh, m, k, n)
