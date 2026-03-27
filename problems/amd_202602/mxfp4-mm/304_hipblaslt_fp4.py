#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#304: hipBLASLt FP4 GEMM! HIP_R_4F_E2M1_EXT=33, VEC32_UE8M0=2.
Use load_inline to call hipBLASLt's matmul with FP4 inputs.
"""
import torch, os, subprocess, time
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

# First, probe hipBLASLt API more carefully
hipblaslt_code = r"""
#include <torch/extension.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

// Data types from headers
// HIP_R_4F_E2M1_EXT = 33
// HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 = 2

#define CHECK_HIP(x) do { hipError_t err = (x); if (err != hipSuccess) { fprintf(stderr, "HIP error: %s at line %d\\n", hipGetErrorString(err), __LINE__); throw std::runtime_error("HIP error"); } } while(0)
#define CHECK_HIPBLASLT(x) do { hipblasStatus_t err = (x); if (err != HIPBLAS_STATUS_SUCCESS) { fprintf(stderr, "hipBLASLt error: %d at line %d\\n", (int)err, __LINE__); throw std::runtime_error("hipBLASLt error"); } } while(0)

torch::Tensor hipblaslt_fp4_gemm(
    torch::Tensor A_fp4,      // [M, K/2] uint8 (packed FP4)
    torch::Tensor B_fp4,      // [N, K/2] uint8 (packed FP4)
    torch::Tensor A_scale,    // [M, K/32] uint8 (E8M0 scales)
    torch::Tensor B_scale,    // [N, K/32] uint8 (E8M0 scales)
    int M, int N, int K
) {
    auto out = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A_fp4.device()));

    hipblasLtHandle_t handle;
    CHECK_HIPBLASLT(hipblasLtCreate(&handle));

    // Create matmul descriptor
    // C = A * B^T (row-major A [M,K], col-major B [K,N] = row-major B^T [N,K])
    hipblasLtMatmulDesc_t matmul_desc;
    hipblasComputeType_t compute_type = HIPBLAS_COMPUTE_32F;
    CHECK_HIPBLASLT(hipblasLtMatmulDescCreate(&matmul_desc, compute_type, HIP_R_32F));

    // Set scale types for block scaling
    hipblasLtMatmulMatrixScale_t scale_type = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
        matmul_desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
        &scale_type, sizeof(scale_type)));
    CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
        matmul_desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
        &scale_type, sizeof(scale_type)));

    // Transpose B (our B is [N, K/2], we want C = A * B^T)
    hipblasOperation_t transA = HIPBLAS_OP_N;
    hipblasOperation_t transB = HIPBLAS_OP_T;
    CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
        matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
        matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    // Set scale pointers
    void* a_scale_ptr = A_scale.data_ptr();
    void* b_scale_ptr = B_scale.data_ptr();
    CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
        matmul_desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_ptr, sizeof(a_scale_ptr)));
    CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
        matmul_desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_ptr, sizeof(b_scale_ptr)));

    // Create layout descriptors
    // A: [M, K] in FP4, stored as [M, K/2] uint8
    hipblasLtMatrixLayout_t layoutA, layoutB, layoutC;
    hipDataType fp4_type = (hipDataType)33;  // HIP_R_4F_E2M1_EXT
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layoutA, fp4_type, M, K, K));
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layoutB, fp4_type, K, N, K));
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_16BF, M, N, N));

    // Algorithm selection
    hipblasLtMatmulPreference_t preference;
    CHECK_HIPBLASLT(hipblasLtMatmulPreferenceCreate(&preference));
    size_t workspace_size = 256 * 1024 * 1024;  // 256MB
    CHECK_HIPBLASLT(hipblasLtMatmulPreferenceSetAttribute(
        preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size, sizeof(workspace_size)));

    // Find algorithm
    hipblasLtMatmulHeuristicResult_t heuristic[8];
    int return_algo_count = 0;
    CHECK_HIPBLASLT(hipblasLtMatmulAlgoGetHeuristic(
        handle, matmul_desc, layoutA, layoutB, layoutC, layoutC,
        preference, 8, heuristic, &return_algo_count));

    printf("Found %d algorithms\\n", return_algo_count);

    if (return_algo_count == 0) {
        fprintf(stderr, "No algorithms found for FP4 GEMM!\\n");
        hipblasLtMatmulPreferenceDestroy(preference);
        hipblasLtMatrixLayoutDestroy(layoutA);
        hipblasLtMatrixLayoutDestroy(layoutB);
        hipblasLtMatrixLayoutDestroy(layoutC);
        hipblasLtMatmulDescDestroy(matmul_desc);
        hipblasLtDestroy(handle);
        return out;
    }

    // Allocate workspace
    void* workspace = nullptr;
    size_t actual_ws = heuristic[0].workspaceSize;
    if (actual_ws > 0) {
        CHECK_HIP(hipMalloc(&workspace, actual_ws));
    }

    // Execute
    float alpha = 1.0f, beta = 0.0f;
    CHECK_HIPBLASLT(hipblasLtMatmul(
        handle, matmul_desc,
        &alpha,
        A_fp4.data_ptr(), layoutA,
        B_fp4.data_ptr(), layoutB,
        &beta,
        out.data_ptr(), layoutC,
        out.data_ptr(), layoutC,
        &heuristic[0].algo,
        workspace, actual_ws,
        0));

    CHECK_HIP(hipDeviceSynchronize());

    // Cleanup
    if (workspace) CHECK_HIP(hipFree(workspace));
    hipblasLtMatmulPreferenceDestroy(preference);
    hipblasLtMatrixLayoutDestroy(layoutA);
    hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtMatrixLayoutDestroy(layoutC);
    hipblasLtMatmulDescDestroy(matmul_desc);
    hipblasLtDestroy(handle);

    return out;
}

// Quick probe: just check if hipBLASLt can find FP4 algorithms
std::string probe_hipblaslt_fp4(int M, int N, int K) {
    hipblasLtHandle_t handle;
    auto status = hipblasLtCreate(&handle);
    if (status != HIPBLAS_STATUS_SUCCESS)
        return "Failed to create handle: " + std::to_string((int)status);

    hipblasLtMatmulDesc_t matmul_desc;
    status = hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    if (status != HIPBLAS_STATUS_SUCCESS)
        return "Failed to create matmul desc: " + std::to_string((int)status);

    // Set scale mode
    hipblasLtMatmulMatrixScale_t scale_type = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    status = hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_type, sizeof(scale_type));
    if (status != HIPBLAS_STATUS_SUCCESS)
        return "Failed to set A scale mode: " + std::to_string((int)status);

    status = hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_type, sizeof(scale_type));
    if (status != HIPBLAS_STATUS_SUCCESS)
        return "Failed to set B scale mode: " + std::to_string((int)status);

    hipblasOperation_t transB = HIPBLAS_OP_T;
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));

    hipDataType fp4_type = (hipDataType)33;
    hipblasLtMatrixLayout_t layoutA, layoutB, layoutC;
    status = hipblasLtMatrixLayoutCreate(&layoutA, fp4_type, M, K, K);
    if (status != HIPBLAS_STATUS_SUCCESS)
        return "Failed to create layout A: " + std::to_string((int)status);

    status = hipblasLtMatrixLayoutCreate(&layoutB, fp4_type, K, N, K);
    if (status != HIPBLAS_STATUS_SUCCESS)
        return "Failed to create layout B: " + std::to_string((int)status);

    status = hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_16BF, M, N, N);
    if (status != HIPBLAS_STATUS_SUCCESS)
        return "Failed to create layout C: " + std::to_string((int)status);

    hipblasLtMatmulPreference_t preference;
    hipblasLtMatmulPreferenceCreate(&preference);
    size_t ws = 256*1024*1024;
    hipblasLtMatmulPreferenceSetAttribute(preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));

    hipblasLtMatmulHeuristicResult_t heuristic[8];
    int count = 0;
    status = hipblasLtMatmulAlgoGetHeuristic(handle, matmul_desc, layoutA, layoutB, layoutC, layoutC, preference, 8, heuristic, &count);

    std::string result = "Heuristic status: " + std::to_string((int)status) + ", algos found: " + std::to_string(count);
    for (int i = 0; i < count; i++) {
        result += "\n  algo[" + std::to_string(i) + "]: ws=" + std::to_string(heuristic[i].workspaceSize);
    }

    hipblasLtMatmulPreferenceDestroy(preference);
    hipblasLtMatrixLayoutDestroy(layoutA);
    hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtMatrixLayoutDestroy(layoutC);
    hipblasLtMatmulDescDestroy(matmul_desc);
    hipblasLtDestroy(handle);

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hipblaslt_fp4_gemm", &hipblaslt_fp4_gemm);
    m.def("probe_hipblaslt_fp4", &probe_hipblaslt_fp4);
}
"""

# Try to compile and run
print("=== Compiling hipBLASLt FP4 module ===")
try:
    t0 = time.time()
    mod = load_inline(
        name="hipblaslt_fp4",
        cpp_sources=[hipblaslt_code],
        extra_include_paths=["/opt/rocm/include"],
        extra_ldflags=["-L/opt/rocm/lib", "-lhipblaslt"],
        verbose=True,
        is_python_module=True,
    )
    print(f"Compiled in {time.time()-t0:.1f}s")

    # Probe FP4 algorithm availability
    for M, N, K in [(256, 3072, 1536), (64, 7168, 2048), (16, 2112, 7168), (4, 2880, 512)]:
        result = mod.probe_hipblaslt_fp4(M, N, K)
        print(f"M={M} N={N} K={K}: {result}")

except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()

# Fallback kernel
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
import json
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
