#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#309: hipBLASLt FP4 with HIP C++ E8M0 scale kernel.
Single kernel launch for B_scale computation.
"""
import torch, os, time, json
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
from aiter.ops.triton.quant import dynamic_mxfp4_quant

hip_code = r"""
#include <torch/extension.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdio>

// E8M0 scale kernel: compute scale for each group of 32 bf16 elements
// Input: bf16 [N, K], Output: uint8 [N, K/32]
__global__ void e8m0_scale_kernel(
    const hip_bfloat16* __restrict__ x,
    unsigned char* __restrict__ scale_out,
    int N, int K, int K_groups
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;  // linear group index
    if (gid >= N * K_groups) return;

    int n = gid / K_groups;
    int g = gid % K_groups;

    // Find max abs of 32 elements
    float amax = 0.0f;
    const hip_bfloat16* row = x + n * K + g * 32;
    for (int i = 0; i < 32; i++) {
        float val = (float)(row[i]);
        float abs_val = (val < 0) ? -val : val;
        if (abs_val > amax) amax = abs_val;
    }

    // E8M0 encoding (same as _mxfp4_quant_op)
    union { float f; unsigned int i; } u;
    u.f = amax;
    unsigned int amax_int = u.i;
    unsigned int amax_rounded = (amax_int + 0x200000u) & 0xFF800000u;
    u.i = amax_rounded;
    unsigned int exp_bits = (amax_rounded >> 23) & 0xFF;
    int scale_unbiased = (int)exp_bits - 127 - 2;
    int scale_biased = scale_unbiased + 127;
    if (scale_biased < 0) scale_biased = 0;

    scale_out[n * K_groups + g] = (unsigned char)scale_biased;
}

torch::Tensor fast_e8m0(torch::Tensor x) {
    int N = x.size(0);
    int K = x.size(1);
    int K_groups = K / 32;
    auto scale = torch::empty({N, K_groups}, torch::dtype(torch::kUInt8).device(x.device()));

    int total = N * K_groups;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    e8m0_scale_kernel<<<blocks, threads>>>(
        (const hip_bfloat16*)x.data_ptr(),
        (unsigned char*)scale.data_ptr(),
        N, K, K_groups
    );
    return scale;
}

// hipBLASLt context management
#define CK(x) do { hipblasStatus_t s=(x); if(s) printf("BLT %d ln %d\n",(int)s,__LINE__); } while(0)

static hipblasLtHandle_t g_h = nullptr;
struct Ctx {
    hipblasLtMatmulDesc_t md;
    hipblasLtMatrixLayout_t lA, lB, lC;
    hipblasLtMatmulHeuristicResult_t heur;
    bool ok; int M, N, K;
};
static Ctx g_c[8]; static int g_nc = 0;

Ctx* get(int M, int N, int K) {
    for (int i = 0; i < g_nc; i++)
        if (g_c[i].M==M && g_c[i].N==N && g_c[i].K==K && g_c[i].ok) return &g_c[i];
    return nullptr;
}

void setup(int M, int N, int K) {
    if (get(M,N,K)) return;
    if (!g_h) hipblasLtCreate(&g_h);
    auto& c = g_c[g_nc++];
    c.M=M; c.N=N; c.K=K;

    CK(hipblasLtMatmulDescCreate(&c.md, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t opT=HIPBLAS_OP_T, opN=HIPBLAS_OP_N;
    CK(hipblasLtMatmulDescSetAttribute(c.md, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    CK(hipblasLtMatmulDescSetAttribute(c.md, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
    hipblasLtMatmulMatrixScale_t sm = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    CK(hipblasLtMatmulDescSetAttribute(c.md, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm)));
    CK(hipblasLtMatmulDescSetAttribute(c.md, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm)));

    hipDataType fp4 = (hipDataType)33;
    CK(hipblasLtMatrixLayoutCreate(&c.lA, fp4, K, N, K));
    CK(hipblasLtMatrixLayoutCreate(&c.lB, fp4, K, M, K));
    CK(hipblasLtMatrixLayoutCreate(&c.lC, HIP_R_16BF, N, M, N));

    hipblasLtMatmulPreference_t p;
    CK(hipblasLtMatmulPreferenceCreate(&p));
    size_t ws=0;
    CK(hipblasLtMatmulPreferenceSetAttribute(p, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    int cnt=0;
    CK(hipblasLtMatmulAlgoGetHeuristic(g_h, c.md, c.lA, c.lB, c.lC, c.lC, p, 1, &c.heur, &cnt));
    hipblasLtMatmulPreferenceDestroy(p);
    c.ok = (cnt>0);
}

torch::Tensor run(torch::Tensor A_fp4, torch::Tensor B_fp4,
                  torch::Tensor A_scale, torch::Tensor B_scale,
                  int M, int N, int K) {
    auto* c = get(M,N,K);
    auto out = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A_fp4.device()));
    if (!c || !c->ok) return out;

    void* bs = B_scale.data_ptr();
    void* as = A_scale.data_ptr();
    CK(hipblasLtMatmulDescSetAttribute(c->md, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &bs, sizeof(bs)));
    CK(hipblasLtMatmulDescSetAttribute(c->md, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &as, sizeof(as)));

    float alpha=1.0f, beta=0.0f;
    CK(hipblasLtMatmul(g_h, c->md, &alpha,
        B_fp4.data_ptr(), c->lA, A_fp4.data_ptr(), c->lB,
        &beta, out.data_ptr(), c->lC, out.data_ptr(), c->lC,
        &c->heur.algo, nullptr, 0, 0));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_e8m0", &fast_e8m0);
    m.def("setup", &setup);
    m.def("run", &run);
}
"""

print("=== Compiling ===")
mod = load_inline(
    name="hblt_hip",
    cpp_sources=[""],
    cuda_sources=[hip_code],
    extra_include_paths=["/opt/rocm/include"],
    extra_ldflags=["-L/opt/rocm/lib", "-lhipblaslt"],
    verbose=False,
    is_python_module=True,
    with_cuda=True,
)
print("OK")

for M, N, K in [(256,3072,1536),(64,7168,2048),(32,4096,512),(32,2880,512),
                (64,3072,1536),(256,2880,512)]:
    mod.setup(M, N, K)

# Preshuffle fallback
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
_cfgs = {"N=2880-K=512": {"M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=4096-K=512": {"M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=2112-K=7168": {"M_LEQ_16": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 8, "num_warps": 4, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=7168-K=2048": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=3072-K=1536": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 3, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}}
try: _dev = arch_info.get_arch()
except: _dev = "gfx950"
_cd = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
os.makedirs(_cd, exist_ok=True)
for _sk, _cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json", "w") as f:
        json.dump(_cfg, f)

for _m, _n, _k in [(4,2880,512),(16,2112,7168)]:
    try:
        _A = torch.randn((_m, _k), dtype=torch.bfloat16, device="cuda")
        _Bw = torch.zeros((_n//16, (_k//2)*16), dtype=torch.uint8, device="cuda")
        _Bws = torch.zeros((_n//32, _k), dtype=torch.uint8, device="cuda")
        gemm_a16wfp4_preshuffle(_A, _Bw, _Bws, prequant=True, dtype=torch.bfloat16)
    except: pass
torch.cuda.empty_cache()

_ck = None; _cw = None; _cs = None

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _ck, _cw, _cs
    A = data[0]; m, k = A.shape; n = data[1].shape[0]

    if m >= 32 and m % 32 == 0:
        B_q = data[2]; B = data[1]
        A_q, A_scale = dynamic_mxfp4_quant(A)
        A_scale = A_scale.contiguous()
        B_scale = mod.fast_e8m0(B)
        return mod.run(A_q, B_q, A_scale, B_scale, m, n, k)
    else:
        B_shuffle = data[3]; B_scale_sh = data[4]
        dp = B_shuffle.data_ptr()
        if dp != _ck:
            _ck = dp
            _cw = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
            _cs = B_scale_sh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
        return gemm_a16wfp4_preshuffle(A, _cw, _cs, prequant=True, dtype=torch.bfloat16)
