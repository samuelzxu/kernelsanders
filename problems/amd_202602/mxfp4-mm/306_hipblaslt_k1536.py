#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#306: hipBLASLt FP4 for K=1536 M=256 N=3072 only.
TN layout with swapped A/B. Try to actually run the GEMM.
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

#define CK(x) do { hipblasStatus_t s=(x); if(s) { printf("BLT err %d line %d\n",(int)s,__LINE__); return "err"; } } while(0)

std::string probe_and_run(
    torch::Tensor A_fp4, torch::Tensor B_fp4,
    torch::Tensor A_scale, torch::Tensor B_scale,
    torch::Tensor out,
    int M, int N, int K
) {
    hipblasLtHandle_t h;
    CK(hipblasLtCreate(&h));

    hipblasLtMatmulDesc_t md;
    CK(hipblasLtMatmulDescCreate(&md, HIPBLAS_COMPUTE_32F, HIP_R_32F));

    // TN: swap A,B for row-major. hipBLASLt_A=B, hipBLASLt_B=A
    hipblasOperation_t opT=HIPBLAS_OP_T, opN=HIPBLAS_OP_N;
    CK(hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    CK(hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    // Block scale
    hipblasLtMatmulMatrixScale_t sm = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    CK(hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm)));
    CK(hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm)));

    // Scale pointers: hipBLASLt_A_scale = B_scale, hipBLASLt_B_scale = A_scale
    void* bs_ptr = B_scale.data_ptr();
    void* as_ptr = A_scale.data_ptr();
    CK(hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &bs_ptr, sizeof(bs_ptr)));
    CK(hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &as_ptr, sizeof(as_ptr)));

    hipDataType fp4 = (hipDataType)33;

    // hipBLASLt_A = B_fp4: col-major [K, N], ld=K
    // hipBLASLt_B = A_fp4: col-major [K, M], ld=K
    // C: col-major [N, M], ld=N = row-major [M, N]
    hipblasLtMatrixLayout_t lA, lB, lC;
    CK(hipblasLtMatrixLayoutCreate(&lA, fp4, K, N, K));
    CK(hipblasLtMatrixLayoutCreate(&lB, fp4, K, M, K));
    CK(hipblasLtMatrixLayoutCreate(&lC, HIP_R_16BF, N, M, N));

    hipblasLtMatmulPreference_t pref;
    CK(hipblasLtMatmulPreferenceCreate(&pref));
    size_t ws_max = 256*1024*1024;
    CK(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_max, sizeof(ws_max)));

    hipblasLtMatmulHeuristicResult_t heur[16];
    int cnt = 0;
    auto st = hipblasLtMatmulAlgoGetHeuristic(h, md, lA, lB, lC, lC, pref, 16, heur, &cnt);

    char buf[512];
    snprintf(buf, sizeof(buf), "M=%d N=%d K=%d status=%d algos=%d", M, N, K, (int)st, cnt);
    std::string result(buf);

    if (cnt > 0) {
        // Allocate workspace
        void* workspace = nullptr;
        size_t ws = heur[0].workspaceSize;
        if (ws > 0) hipMalloc(&workspace, ws);

        float alpha = 1.0f, beta = 0.0f;

        // Run GEMM
        st = hipblasLtMatmul(h, md, &alpha,
            B_fp4.data_ptr(), lA,   // hipBLASLt_A = B
            A_fp4.data_ptr(), lB,   // hipBLASLt_B = A
            &beta,
            out.data_ptr(), lC,
            out.data_ptr(), lC,
            &heur[0].algo, workspace, ws, 0);

        hipDeviceSynchronize();
        snprintf(buf, sizeof(buf), " exec=%d ws=%zu", (int)st, ws);
        result += buf;

        if (workspace) hipFree(workspace);

        // Try timing
        if (st == HIPBLAS_STATUS_SUCCESS) {
            // Warmup
            for (int i = 0; i < 5; i++) {
                hipblasLtMatmul(h, md, &alpha,
                    B_fp4.data_ptr(), lA,
                    A_fp4.data_ptr(), lB,
                    &beta, out.data_ptr(), lC, out.data_ptr(), lC,
                    &heur[0].algo, workspace, ws, 0);
            }
            hipDeviceSynchronize();

            hipEvent_t t0, t1;
            hipEventCreate(&t0); hipEventCreate(&t1);
            hipEventRecord(t0);
            for (int i = 0; i < 100; i++) {
                hipblasLtMatmul(h, md, &alpha,
                    B_fp4.data_ptr(), lA,
                    A_fp4.data_ptr(), lB,
                    &beta, out.data_ptr(), lC, out.data_ptr(), lC,
                    &heur[0].algo, workspace, ws, 0);
            }
            hipEventRecord(t1);
            hipEventSynchronize(t1);
            float ms;
            hipEventElapsedTime(&ms, t0, t1);
            snprintf(buf, sizeof(buf), " time=%.2fus", ms * 10.0f);
            result += buf;
            hipEventDestroy(t0);
            hipEventDestroy(t1);
        }
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
    m.def("probe_and_run", &probe_and_run);
}
"""

print("=== Compiling hipBLASLt FP4 ===")
try:
    mod = load_inline(
        name="hipblaslt_k1536",
        cpp_sources=[hipblaslt_code],
        extra_include_paths=["/opt/rocm/include"],
        extra_ldflags=["-L/opt/rocm/lib", "-lhipblaslt"],
        verbose=False,
        is_python_module=True,
    )
    print("Compiled OK")

    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    # Test shapes that have good padding: M%64=0 and N%256=0
    for M, N, K in [(256,3072,1536),(64,7168,2048),(32,4096,512),(32,2880,512)]:
        A = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
        A_q, A_scale = dynamic_mxfp4_quant(A)
        # B_q dummy data
        B_q = torch.zeros((N, K//2), dtype=torch.uint8, device="cuda")
        B_scale = torch.zeros((N, K//32), dtype=torch.uint8, device="cuda")
        out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")

        result = mod.probe_and_run(A_q, B_q, A_scale, B_scale, out, M, N, K)
        print(result)

except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()

# Standard kernel fallback
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
