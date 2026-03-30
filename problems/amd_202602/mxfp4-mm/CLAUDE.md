# MXFP4-MM Project Guide

## Overview
FP4 GEMM kernel optimization for AMD MI355X (gfx950) — GPU MODE competition.
6 benchmark shapes (M=4-256, K=512-7168, N=2880-7168). Score = geometric mean of per-shape times.

## Key Files
- `submission.py` — current leaderboard submission (#591, ~9.8µs geomean)
- `docker/` — cross-compiled HIP kernels (.hip source, .so binaries)
- `PROGRESS.md` — full experiment history (600+ experiments)
- `CONTEXT.md` — optimization context and findings
- `.claude/memory.md` — session memory with technical details

## Cross-Docker HIP Compilation Workflow

We don't have an MI355X GPU locally. HIP kernels must be cross-compiled for gfx950 using Docker, then JIT-compiled on the runner via `load_inline`.

### Step 1: Write the kernel

Create a `.hip` file in `docker/`:
```cpp
// docker/my_kernel.hip
#include <hip/hip_runtime.h>
#include <torch/extension.h>

extern "C" __global__ __launch_bounds__(256, 2)
void my_kernel(const unsigned short* A, unsigned short* C, int M, int N) {
    // kernel code using __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4 etc.
}

// PyTorch wrapper
torch::Tensor run_my_kernel(torch::Tensor A, int N) {
    auto C = torch::empty({(int)A.size(0), N}, torch::dtype(torch::kBFloat16).device(A.device()));
    int grid = ...;
    hipLaunchKernelGGL(my_kernel, dim3(grid), dim3(256), 0, 0,
        (const unsigned short*)A.data_ptr(), (unsigned short*)C.data_ptr(),
        (int)A.size(0), N);
    return C;
}
```

**Important**: Use `hipLaunchKernelGGL()` instead of `<<<>>>` syntax. The `<<<>>>` gets mangled by hipcc when embedded in Python string literals for `load_inline`.

### Step 2: Cross-compile locally to verify it compiles

```bash
docker run --rm -v $(pwd)/docker:/workspace rocm/dev-ubuntu-24.04:7.1-complete bash -c "
pip install --break-system-packages torch==2.10.0+rocm7.1 --index-url https://download.pytorch.org/whl/rocm7.1 2>&1 | tail -1
TORCH_DIR=\$(python3 -c 'import torch; print(torch.__path__[0])')
cd /workspace
/opt/rocm/bin/hipcc -O3 -std=c++20 --offload-arch=gfx950 -mcumode \
    -I\$TORCH_DIR/include -I\$TORCH_DIR/include/torch/csrc/api/include \
    -isystem /usr/include/python3.12 \
    -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 \
    -D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1 \
    -DTORCH_EXTENSION_NAME=my_ext \
    -shared -fPIC \
    -L\$TORCH_DIR/lib -ltorch -ltorch_cpu -lc10 -lc10_hip -ltorch_hip \
    -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,\$TORCH_DIR/lib \
    -o my_kernel.so my_kernel.hip 2>&1
ls -la my_kernel.so && echo SUCCESS || echo FAILED
"
```

This only checks compilation — the `.so` won't run locally (no gfx950 GPU).

### Step 3: Embed in submission via load_inline

The runner HAS the GPU and matching ROCm. Use `load_inline` to JIT-compile on the runner:

```python
from torch.utils.cpp_extension import load_inline
import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'

_hip_src = r"""
... paste the .hip source here as a raw string ...
"""

_cpp_src = r"""
#include <torch/extension.h>
torch::Tensor run_my_kernel(torch::Tensor A, int N);
"""

mod = load_inline(
    name='my_ext',
    cpp_sources=_cpp_src,
    cuda_sources=_hip_src,
    functions=['run_my_kernel'],
    verbose=False,
    extra_cuda_cflags=['-O3', '-w', '-mcumode', '--offload-arch=gfx950'],
)

# Now callable:
result = mod.run_my_kernel(A_tensor, N)
```

### Step 4: Submit and test

```bash
popcorn submit --mode test --no-tui my_submission.py    # correctness
popcorn submit --mode benchmark --no-tui my_submission.py  # timing
popcorn submit --mode profile --no-tui my_submission.py    # GPU profiling
popcorn submit --mode leaderboard --no-tui my_submission.py  # ranked
```

### Common Pitfalls

1. **Banned word**: The server rejects submissions containing the word "stream" (case-sensitive). Use `0` for stream args in C++, and avoid the word in comments.

2. **`<<<>>>` syntax**: Gets mangled when hipcc processes the source from `load_inline`. Always use `hipLaunchKernelGGL(kernel, dim3(grid), dim3(block), shared, 0, args...)` instead.

3. **Rate limiting**: 6 submissions per hour per mode. Parallel submissions all count against the limit. Submit sequentially.

4. **JIT compilation time**: `load_inline` takes ~15-45 seconds on the runner. This happens at import time, not during benchmark.

5. **CUDA event timing**: The benchmark measures GPU-side time only (between `start_event.record()` and `end_event.record()`). CPU dispatch overhead is NOT measured. Only GPU kernel execution time matters.

6. **LDS sizing**: K_STEP FP4 packed bytes ≠ K_STEP bf16 bytes. For K_STEP=128 FP4 bytes, the bf16 source is 256 values × 2 bytes = 512 bytes per row. Don't halve the LDS allocation.

7. **Preshuffle B address formula**: `Bw[sr * bw_stride + kb*512 + kh*256 + nw*16 + ki]` where `sr=n/16, nw=n%16, kb=k_byte/32, kh=(k_byte%32)/16, ki=k_byte%16`. But this per-byte formula is SLOW — use contiguous loads and rearrange in LDS instead.

8. **A LDS bank conflicts**: Stride between lanes should NOT be a multiple of 256 bytes (64 banks × 4 bytes). Pad rows by 16 bytes (e.g., 512→528 stride).

## Available MFMA Instructions (gfx950 FP4)

```cpp
// 32×32 output, K=64 FP4, 32 cycles
v16f acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
    a_v8i, b_v8i, acc_v16f, /*Atype=*/4, /*Btype=*/4,
    /*opsel_a=*/0, scale_a_u32, /*opsel_b=*/0, scale_b_u32);

// 16×16 output, K=128 FP4, 16 cycles (Triton uses this one)
v4f acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
    a_v8i, b_v8i, acc_v4f, 4, 4, 0, scale_a, 0, scale_b);
// Use op_sel:[1,1,0] for chaining: processes upper 32 bytes of A/B
```

Scale packing (v3): `pas = scale_byte | (scale_byte << 8)` — broadcast own scale to both bytes.

## Triton ASM Findings (probe 595)

The preshuffle Triton kernel (BSM=32 BSN=32 K=512) generates:
- 8 `v_mfma_scale_f32_16x16x128_f8f6f4` calls (NOT 32×32×64)
- 1505 VALU ops (inline A quantization — branchless)
- 11 global loads, 15 ds_reads, 9 ds_writes
- Uses `op_sel:[1,1,0]` to chain MFMAs
- 0.479µs GPU time per shape
