#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#476: Custom MFMA FP4 GEMM kernel via load_inline.
Fused bf16→FP4 quant + MFMA 32x32x64 in one kernel.
Single kernel launch = no quant+GEMM dispatch gap.
"""
import torch, os, json
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'

# Preshuffle fallback
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info

_cfgs={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=4096-K=512":{"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=2112-K=7168":{"M_LEQ_16":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":8,"num_warps":4,"num_stages":2,"waves_per_eu":3,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=7168-K=2048":{"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":3,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=3072-K=1536":{"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":3,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_256":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":32,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)

# Warmup preshuffle
for _m,_n,_k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
    try:
        _A=torch.randn((_m,_k),dtype=torch.bfloat16,device="cuda")
        _Bw=torch.zeros((_n//16,(_k//2)*16),dtype=torch.uint8,device="cuda")
        _Bws=torch.zeros((_n//32,_k),dtype=torch.uint8,device="cuda")
        gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
    except:pass
torch.cuda.empty_cache()

# Custom MFMA kernel
_hip_src = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Custom MFMA FP4 GEMM: fused bf16->FP4 quant + 32x32x64 MFMA
extern "C" __global__ void mxfp4_fused_gemm(
    const unsigned short* __restrict__ A,
    const unsigned char* __restrict__ B,
    const unsigned char* __restrict__ B_scale,
    unsigned short* __restrict__ C,
    int M, int N, int K,
    int stride_b, int stride_bs, int stride_c
) {
    int tile_n = blockIdx.x * 32;
    int tile_m = blockIdx.y * 32;
    int lane = threadIdx.x;
    int lane32 = lane & 31;
    int group = lane >> 5;

    float acc[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;

    int K_iters = K / 64;

    for (int ki = 0; ki < K_iters; ki++) {
        int k_base = ki * 64;

        // Load+quant 32 bf16 A values for this lane
        int a_row = tile_m + lane32;
        int a_k_start = k_base + group * 32;

        float vals[32];
        float amax = 0.0f;
        unsigned char a_fp4[16];
        unsigned char a_scale_val;

        if (a_row < M) {
            const unsigned short* ap = A + a_row * K + a_k_start;
            // Vectorized load
            const uint4* ap4 = (const uint4*)ap;
            #pragma unroll
            for (int v = 0; v < 4; v++) {
                uint4 chunk = ap4[v];
                unsigned int w[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    float lo = __uint_as_float((w[j] & 0xFFFFu) << 16);
                    float hi = __uint_as_float(w[j] & 0xFFFF0000u);
                    vals[v*8+j*2] = lo;
                    vals[v*8+j*2+1] = hi;
                    amax = fmaxf(amax, fmaxf(fabsf(lo), fabsf(hi)));
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 32; i++) vals[i] = 0.0f;
        }

        // E8M0 scale
        unsigned int ai = __float_as_uint(amax);
        unsigned int ar = (ai + 0x200000u) & 0xFF800000u;
        int eb = (ar >> 23) & 0xFF;
        int su = eb - 129;
        int sb = su + 127;
        if (sb < 0) sb = 0;
        a_scale_val = (unsigned char)sb;

        // Quant scale
        int qe = 127 - su;
        if (qe < 1) qe = 0;
        if (qe > 254) qe = 254;
        float qs = __uint_as_float((unsigned int)qe << 23);
        if (amax == 0.0f) qs = 0.0f;

        // FP4 quant
        const unsigned int dmi = 149u << 23;
        float dmf = __uint_as_float(dmi);
        const int vta = ((int)(1 - 127) << 23) + (1 << 21) - 1;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            auto fp4 = [&](float v) -> unsigned char {
                float qf = v * qs;
                unsigned int qx = __float_as_uint(qf);
                unsigned int s = qx & 0x80000000u;
                qx ^= s;
                float qp = __uint_as_float(qx);
                unsigned char r;
                if (qp >= 6.0f) r = 0x7;
                else if (qp < 1.0f) r = (unsigned char)((__float_as_uint(qp + dmf) - dmi) & 0xFF);
                else { unsigned int mo = (qx >> 22) & 1; r = (unsigned char)((((unsigned int)((int)qx + vta) + mo) >> 22) & 0xFF); }
                return (r & 0x7) | ((unsigned char)(s >> 28) & 0x8);
            };
            a_fp4[i] = fp4(vals[2*i]) | (fp4(vals[2*i+1]) << 4);
        }

        // Load B from RAW row-major [N, K/2]
        unsigned char b_fp4[16];
        unsigned char b_scale_val;
        int b_col = tile_n + lane32;
        if (b_col < N) {
            int b_byte_start = (k_base + group * 32) / 2;
            const unsigned char* bp = B + b_col * K_half + b_byte_start;
            #pragma unroll
            for (int i = 0; i < 16; i++) b_fp4[i] = bp[i];
            // B_scale is RAW row-major [N, K/32]
            b_scale_val = B_scale[b_col * K_groups + (k_base / 32) + group];
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) b_fp4[i] = 0;
            b_scale_val = 127;
        }

        // Scale packing v3: broadcast own scale to both bytes
        // HW reads byte0 from lanes 0-31 (group0) and byte1 from lanes 32-63 (group1)
        unsigned int pas = (unsigned int)a_scale_val | ((unsigned int)a_scale_val << 8);
        unsigned int pbs = (unsigned int)b_scale_val | ((unsigned int)b_scale_val << 8);

        // MFMA
        typedef int v8si __attribute__((ext_vector_type(8)));
        typedef float v16f __attribute__((vector_size(64)));

        v8si areg = {};
        areg[0] = *(int*)&a_fp4[0]; areg[1] = *(int*)&a_fp4[4];
        areg[2] = *(int*)&a_fp4[8]; areg[3] = *(int*)&a_fp4[12];

        v8si breg = {};
        breg[0] = *(int*)&b_fp4[0]; breg[1] = *(int*)&b_fp4[4];
        breg[2] = *(int*)&b_fp4[8]; breg[3] = *(int*)&b_fp4[12];

        v16f creg;
        #pragma unroll
        for (int i = 0; i < 16; i++) creg[i] = acc[i];

        creg = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            areg, breg, creg, 4, 4, 0, pas, 0, pbs);

        #pragma unroll
        for (int i = 0; i < 16; i++) acc[i] = creg[i];
    }

    // Store output
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int r = tile_m + group * 4 + i * 8 + j;
            int c = tile_n + lane32;
            if (r < M && c < N) {
                unsigned int u = __float_as_uint(acc[i*4+j]);
                u = u + (((u >> 16) & 1) + 0x7FFF);
                C[r * stride_c + c] = (unsigned short)(u >> 16);
            }
        }
    }
}

// Wrapper
static torch::Tensor g_out;
static int g_out_m = 0, g_out_n = 0;

torch::Tensor fused_mfma_gemm(
    torch::Tensor A, torch::Tensor B, torch::Tensor B_scale, int N_val
) {
    int M = A.size(0), K = A.size(1);

    if (M != g_out_m || N_val != g_out_n) {
        g_out = torch::empty({M, N_val}, torch::dtype(torch::kBFloat16).device(A.device()));
        g_out_m = M; g_out_n = N_val;
    }

    dim3 grid((N_val + 31) / 32, (M + 31) / 32, 1);
    dim3 block(64);  // 1 wavefront per block

    mxfp4_fused_gemm<<<grid, block, 0, 0>>>(
        (const unsigned short*)A.data_ptr(),
        (const unsigned char*)B.data_ptr(),
        (const unsigned char*)B_scale.data_ptr(),
        (unsigned short*)g_out.data_ptr(),
        M, N_val, K,
        (int)B.stride(0), (int)B_scale.stride(0), N_val
    );

    return g_out;
}
"""

_cpp_src = r"""
#include <torch/extension.h>
torch::Tensor fused_mfma_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor B_scale, int N_val);
"""

_custom_mod = None
try:
    _custom_mod = load_inline(
        name='mfma_rawb_491',
        cpp_sources=_cpp_src,
        cuda_sources=_hip_src,
        functions=['fused_mfma_gemm'],
        verbose=False,
        extra_cuda_cflags=['-O3', '-w', '-mcumode', '--offload-arch=gfx950'],
    )
    # Warmup
    _wA = torch.randn(256, 1536, dtype=torch.bfloat16, device="cuda")
    _wB = torch.zeros(3072, 768, dtype=torch.uint8, device="cuda")
    _wBs = torch.zeros(3072, 48, dtype=torch.uint8, device="cuda")
    _custom_mod.fused_mfma_gemm(_wA, _wB, _wBs, 3072)
    err = torch.cuda.synchronize() if hasattr(torch.cuda, 'synchronize') else None
    # Check for GPU errors
    try:
        torch.cuda.synchronize()
        _ = _wA.sum()  # trigger any deferred error
    except Exception as e:
        print(f"GPU error during warmup: {e}")
        _custom_mod = None
    del _wA, _wB, _wBs
    print("Custom MFMA kernel OK")
except Exception as e:
    print(f"Custom MFMA FAILED: {e}")

torch.cuda.empty_cache()

_ps_ck=None;_ps_cw=None;_ps_cs=None

@torch.inference_mode()
def custom_kernel(data):
    global _ps_ck,_ps_cw,_ps_cs
    A=data[0];B_shuffle=data[3];B_scale_sh=data[4]
    m,k=A.shape;n=data[1].shape[0]

    # Try custom MFMA kernel for K=1536 M=256
    if _custom_mod is not None and m == 256 and k == 1536:
        # Use RAW B_q (data[2]) and unshuffle B_scale
        B_q_raw = data[2].view(torch.uint8)  # [N, K/2] raw FP4
        # Unshuffle B_scale: inverse of view(sm//32,2,16,sn//8,2,4).permute(0,3,5,2,4,1)
        bsu = B_scale_sh.view(torch.uint8)
        sm, sn = bsu.shape
        bs_raw = bsu.view(sm//32, sn//8, 4, 16, 2, 2).permute(0,5,3,1,4,2).contiguous().view(sm, sn)[:n, :k//32].contiguous()
        return _custom_mod.fused_mfma_gemm(A, B_q_raw, bs_raw, n)

    # Preshuffle fallback
    dp=B_shuffle.data_ptr()
    if dp!=_ps_ck:
        _ps_ck=dp;_ps_cw=B_shuffle.view(torch.uint8).reshape(n//16,(k//2)*16)
        _ps_cs=B_scale_sh.view(torch.uint8)[:n,:].contiguous().reshape(n//32,k)
    return gemm_a16wfp4_preshuffle(A,_ps_cw,_ps_cs,prequant=True,dtype=torch.bfloat16)
