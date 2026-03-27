#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#313: hipBLASLt FP4 with e8m0_unshuffle (reverse permutation).
The shuffle is: view(sm//32, 2, 16, sn//8, 2, 4).permute(0, 3, 5, 2, 4, 1)
Inverse permutation of (0,3,5,2,4,1) is (0,5,3,1,4,2).
"""
import torch, os, json
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
from aiter.ops.triton.quant import dynamic_mxfp4_quant

def e8m0_unshuffle(scale_sh, N, K_groups):
    """Reverse e8m0_shuffle. Input is padded [N_pad, K_groups_pad], output is [N, K_groups]."""
    sm, sn = scale_sh.shape
    # Reverse the view and permute
    # Forward: view(sm//32, 2, 16, sn//8, 2, 4).permute(0, 3, 5, 2, 4, 1)
    # The output shape after permute is [sm//32, sn//8, 4, 16, 2, 2]
    # To unshuffle: view as [sm//32, sn//8, 4, 16, 2, 2], apply inverse permute, reshape
    t = scale_sh.view(sm // 32, sn // 8, 4, 16, 2, 2)
    # Inverse of permute(0, 3, 5, 2, 4, 1) is permute(0, 5, 3, 1, 4, 2)
    t = t.permute(0, 5, 3, 1, 4, 2).contiguous()
    t = t.view(sm, sn)
    return t[:N, :K_groups].contiguous()

hipblaslt_code = r"""
#include <torch/extension.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hip/hip_runtime.h>
#include <cstdio>

#define CK(x) do { hipblasStatus_t s=(x); if(s) printf("E%d@%d\n",(int)s,__LINE__); } while(0)

static hipblasLtHandle_t g_h = nullptr;
struct Ctx { hipblasLtMatmulDesc_t md; hipblasLtMatrixLayout_t lA,lB,lC; hipblasLtMatmulHeuristicResult_t heur; bool ok; int M,N,K; };
static Ctx g_c[8]; static int g_nc = 0;

Ctx* get(int M, int N, int K) { for(int i=0;i<g_nc;i++) if(g_c[i].M==M&&g_c[i].N==N&&g_c[i].K==K&&g_c[i].ok) return &g_c[i]; return nullptr; }

void setup(int M, int N, int K) {
    if(get(M,N,K)) return;
    if(!g_h) hipblasLtCreate(&g_h);
    auto& c=g_c[g_nc++]; c.M=M; c.N=N; c.K=K;
    CK(hipblasLtMatmulDescCreate(&c.md, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t opT=HIPBLAS_OP_T, opN=HIPBLAS_OP_N;
    CK(hipblasLtMatmulDescSetAttribute(c.md, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    CK(hipblasLtMatmulDescSetAttribute(c.md, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
    hipblasLtMatmulMatrixScale_t sm = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    CK(hipblasLtMatmulDescSetAttribute(c.md, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm)));
    CK(hipblasLtMatmulDescSetAttribute(c.md, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm)));
    hipDataType fp4=(hipDataType)33;
    CK(hipblasLtMatrixLayoutCreate(&c.lA, fp4, K, N, K));
    CK(hipblasLtMatrixLayoutCreate(&c.lB, fp4, K, M, K));
    CK(hipblasLtMatrixLayoutCreate(&c.lC, HIP_R_16BF, N, M, N));
    hipblasLtMatmulPreference_t p; CK(hipblasLtMatmulPreferenceCreate(&p));
    size_t ws=0; CK(hipblasLtMatmulPreferenceSetAttribute(p, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    int cnt=0; CK(hipblasLtMatmulAlgoGetHeuristic(g_h, c.md, c.lA, c.lB, c.lC, c.lC, p, 1, &c.heur, &cnt));
    hipblasLtMatmulPreferenceDestroy(p); c.ok=(cnt>0);
}

torch::Tensor run(torch::Tensor A_fp4, torch::Tensor B_fp4, torch::Tensor A_scale, torch::Tensor B_scale, int M, int N, int K) {
    auto* c=get(M,N,K);
    auto out=torch::empty({M,N}, torch::dtype(torch::kBFloat16).device(A_fp4.device()));
    if(!c||!c->ok) return out;
    void* bs=B_scale.data_ptr(); void* as=A_scale.data_ptr();
    CK(hipblasLtMatmulDescSetAttribute(c->md, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &bs, sizeof(bs)));
    CK(hipblasLtMatmulDescSetAttribute(c->md, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &as, sizeof(as)));
    float alpha=1.0f, beta=0.0f;
    CK(hipblasLtMatmul(g_h, c->md, &alpha, B_fp4.data_ptr(), c->lA, A_fp4.data_ptr(), c->lB, &beta, out.data_ptr(), c->lC, out.data_ptr(), c->lC, &c->heur.algo, nullptr, 0, 0));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("setup", &setup); m.def("run", &run); }
"""

mod = load_inline(name="hblt_us", cpp_sources=[hipblaslt_code], extra_include_paths=["/opt/rocm/include"],
                  extra_ldflags=["-L/opt/rocm/lib", "-lhipblaslt"], verbose=False, is_python_module=True)

for M,N,K in [(256,3072,1536),(64,7168,2048),(32,4096,512),(32,2880,512),(64,3072,1536),(256,2880,512)]:
    mod.setup(M, N, K)

# Preshuffle
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

for _m, _n, _k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048)]:
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
        # hipBLASLt path with unshuffle
        B_q = data[2]  # raw FP4 [N, K//2]
        B_scale_sh = data[4]  # shuffled E8M0

        # Quant A
        A_q, A_scale = dynamic_mxfp4_quant(A)
        A_scale = A_scale.contiguous()

        # Unshuffle B_scale
        B_scale = e8m0_unshuffle(B_scale_sh, n, k // 32)

        return mod.run(A_q, B_q, A_scale, B_scale, m, n, k)
    else:
        B_shuffle = data[3]; B_scale_sh = data[4]
        dp = B_shuffle.data_ptr()
        if dp != _ck:
            _ck = dp
            _cw = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
            _cs = B_scale_sh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
        return gemm_a16wfp4_preshuffle(A, _cw, _cs, prequant=True, dtype=torch.bfloat16)
