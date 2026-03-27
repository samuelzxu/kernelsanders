#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#305: hipBLASLt FP4 GEMM with TN layout + M padding.
FP4 kernels need: transA=T, transB=N, M%32=0.
Row-major A[M,K] = col-major A^T[K,M]. Row-major B[N,K] = col-major B^T[K,N].
So TN: A^T * B^T = nope... need to think about this more carefully.

Actually: C[M,N] = A[M,K] * B[K,N]
Row-major A[M,K] → col-major view is [K,M] → transA=T means A^T[M,K]
Row-major B_q[N,K/2] → col-major view is [K/2,N] → but we want B[K,N]

For C = A @ B^T where A is [M,K] and B is [N,K]:
In col-major: C[N,M] = B * A^T → transA_blas=N, transB_blas=T
But the scale restriction says "TN only".

Alternative: compute C^T = B @ A^T then transpose output.
C^T[N,M] = B[N,K] * A^T[K,M]
Row-major B[N,K] = col-major [K,N] → transB=N (it's already K×N col-major)
Row-major A[M,K] = col-major [K,M] → transA=N (it's already K×M col-major)
Hmm that gives NN layout.

Actually let me think of it as: we want C[M,N] = A[M,K] @ B[K,N].
Our B is stored as B_q[N, K/2] row-major (packed FP4).
To get B[K,N], we'd store B_q as [K/2, N] col-major.
But our B_q IS [N, K/2] row-major = [K/2, N] col-major.

So in col-major: A is [K,M] and B is [K/2,N] (which represents [K,N] in FP4).
C = A^T * B → TN: transA=T, transB=N.
A layout: [K, M], ld=K
B layout: [K, N], ld=K
C layout: [M, N], ld=M

Wait but C output in col-major [M,N] with ld=M means column-major output.
We need row-major output [M,N] with ld=N.

Row-major C[M,N] = col-major C^T[N,M].
So we'd tell hipBLASLt output is [N,M] with ld=N.

Hmm, this is getting confusing. Let me just do it the standard way:
BLAS convention: C = alpha * op(A) * op(B) + beta * C
All matrices in column-major.
op(A) = A^T if transA = T.

We want: C_rm[M,N] = A_rm[M,K] * B_rm[N,K]^T

In BLAS column-major:
C_cm[N,M] = B_cm[N,K] * A_cm[K,M]
This is: transA=N, transB=N (NN layout).

But FP4 only supports TN. So let's reformulate:
C_cm[N,M] = op_a(A_store) * op_b(B_store)

For TN: op_a = T, op_b = N
C_cm[N,M] = A_store^T * B_store
Need: A_store^T = B_cm[N,K], so A_store = B_cm^T = [K,N] col-major
And B_store = A_cm[K,M]

So: pass B data as "A" to hipBLASLt (transA=T, stored [K,N] col-major)
    pass A data as "B" to hipBLASLt (transB=N, stored [K,M] col-major)

Row-major B_q[N,K/2] = col-major [K/2,N] = FP4 [K,N] col-major ✓
Row-major A_fp4[M,K/2] = col-major [K/2,M] = FP4 [K,M] col-major ✓

So: hipBLASLt_A = B_fp4 data, layout (K,N,K), transA=T → gives [N,K]
    hipBLASLt_B = A_fp4 data, layout (K,M,K), transB=N → gives [K,M]
    C = [N,M] with ld=N (col-major [N,M] = row-major [M,N])

Output layout: (N,M,N) which is col-major [N,M] = row-major C[M,N] ✓
"""
import torch, os, time, json
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

hipblaslt_code = r"""
#include <torch/extension.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

#define CHECK_HIP(x) do { hipError_t err = (x); if (err != hipSuccess) { fprintf(stderr, "HIP err %d at %d\n", (int)err, __LINE__); throw std::runtime_error("HIP err"); } } while(0)
#define CHECK_BLT(x) do { hipblasStatus_t err = (x); if (err != HIPBLAS_STATUS_SUCCESS) { fprintf(stderr, "BLT err %d at %d\n", (int)err, __LINE__); throw std::runtime_error("BLT err"); } } while(0)

// Probe: find algorithms for given shape
std::string probe_fp4(int M, int N, int K) {
    hipblasLtHandle_t h;
    CHECK_BLT(hipblasLtCreate(&h));

    hipblasLtMatmulDesc_t md;
    CHECK_BLT(hipblasLtMatmulDescCreate(&md, HIPBLAS_COMPUTE_32F, HIP_R_32F));

    // TN layout: swap A and B for row-major→col-major conversion
    // hipBLASLt_A = our B_fp4, transA=T
    // hipBLASLt_B = our A_fp4, transB=N
    hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    CHECK_BLT(hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    CHECK_BLT(hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    // Block scale mode
    hipblasLtMatmulMatrixScale_t sm = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    CHECK_BLT(hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm)));
    CHECK_BLT(hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm)));

    hipDataType fp4 = (hipDataType)33;  // HIP_R_4F_E2M1_EXT

    // hipBLASLt_A = B_fp4: stored col-major [K, N], ld=K
    // hipBLASLt_B = A_fp4: stored col-major [K, M_padded], ld=K
    int M_pad = ((M + 31) / 32) * 32;

    hipblasLtMatrixLayout_t lA, lB, lC;
    CHECK_BLT(hipblasLtMatrixLayoutCreate(&lA, fp4, K, N, K));      // B data
    CHECK_BLT(hipblasLtMatrixLayoutCreate(&lB, fp4, K, M_pad, K));  // A data (padded)
    CHECK_BLT(hipblasLtMatrixLayoutCreate(&lC, HIP_R_16BF, N, M_pad, N));  // output

    hipblasLtMatmulPreference_t pref;
    CHECK_BLT(hipblasLtMatmulPreferenceCreate(&pref));
    size_t ws = 256*1024*1024;
    CHECK_BLT(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));

    hipblasLtMatmulHeuristicResult_t heur[16];
    int cnt = 0;
    auto st = hipblasLtMatmulAlgoGetHeuristic(h, md, lA, lB, lC, lC, pref, 16, heur, &cnt);

    std::string result = "M=" + std::to_string(M) + "(pad=" + std::to_string(M_pad) +
        ") N=" + std::to_string(N) + " K=" + std::to_string(K) +
        " status=" + std::to_string((int)st) + " algos=" + std::to_string(cnt);

    for (int i = 0; i < cnt && i < 5; i++) {
        result += "\n  algo[" + std::to_string(i) + "]: ws=" + std::to_string(heur[i].workspaceSize);
    }

    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatrixLayoutDestroy(lA);
    hipblasLtMatrixLayoutDestroy(lB);
    hipblasLtMatrixLayoutDestroy(lC);
    hipblasLtMatmulDescDestroy(md);
    hipblasLtDestroy(h);
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("probe_fp4", &probe_fp4);
}
"""

print("=== Compiling ===")
try:
    t0 = time.time()
    mod = load_inline(
        name="hipblaslt_tn",
        cpp_sources=[hipblaslt_code],
        extra_include_paths=["/opt/rocm/include"],
        extra_ldflags=["-L/opt/rocm/lib", "-lhipblaslt"],
        verbose=False,
        is_python_module=True,
    )
    print(f"Compiled in {time.time()-t0:.1f}s")

    # Probe all benchmark shapes
    for M, N, K in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
        result = mod.probe_fp4(M, N, K)
        print(result)

except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()

# Fallback kernel
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

for _m, _n, _k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
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
    A = data[0]; B_shuffle = data[3]; B_scale_sh = data[4]
    m, k = A.shape; n = data[1].shape[0]
    dp = B_shuffle.data_ptr()
    if dp != _ck:
        _ck = dp
        _cw = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        _cs = B_scale_sh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
    return gemm_a16wfp4_preshuffle(A, _cw, _cs, prequant=True, dtype=torch.bfloat16)
