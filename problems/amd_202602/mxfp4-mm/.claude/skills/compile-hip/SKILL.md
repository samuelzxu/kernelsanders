---
name: compile-hip
description: Cross-compile a HIP kernel for gfx950 (MI355X) using Docker, then build a popcorn submission that JIT-compiles it on the runner via load_inline.
disable-model-invocation: true
argument-hint: <kernel.hip> [submission_number]
---

# Cross-Docker HIP Compilation

Compile a HIP kernel for AMD MI355X (gfx950) and package it for the popcorn competition runner.

**Input**: $ARGUMENTS
- First arg: path to .hip file (e.g., `docker/fp4_v7.hip`)
- Second arg (optional): submission number (e.g., `607`)

## Step 1: Validate the kernel source

Read the .hip file. Check for:
- `hipLaunchKernelGGL` (NOT `<<<>>>` — that gets mangled by hipcc in load_inline strings)
- The word "stream" must NOT appear anywhere (banned by competition server)
- `__launch_bounds__` is specified
- A `torch::Tensor run_*()` wrapper function exists
- No `TORCH_LIBRARY` (load_inline uses pybind11 `functions=[]` instead)

If any issues found, fix them before proceeding.

## Step 2: Cross-compile in Docker

Run this to verify the kernel compiles for gfx950:

```bash
docker run --rm \
  -v $(pwd)/docker:/workspace \
  rocm/dev-ubuntu-24.04:7.1-complete bash -c "
pip install --break-system-packages torch==2.10.0+rocm7.1 --index-url https://download.pytorch.org/whl/rocm7.1 2>&1 | tail -1
TORCH_DIR=\$(python3 -c 'import torch; print(torch.__path__[0])')
cd /workspace
/opt/rocm/bin/hipcc -O3 -std=c++20 --offload-arch=gfx950 -mcumode \
    -I\$TORCH_DIR/include -I\$TORCH_DIR/include/torch/csrc/api/include \
    -isystem /usr/include/python3.12 \
    -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 \
    -D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1 \
    -DTORCH_EXTENSION_NAME=kernel_ext \
    -shared -fPIC \
    -L\$TORCH_DIR/lib -ltorch -ltorch_cpu -lc10 -lc10_hip -ltorch_hip \
    -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,\$TORCH_DIR/lib \
    -o OUTPUT.so INPUT.hip 2>&1
ls -la OUTPUT.so && echo SUCCESS || echo FAILED
"
```

Replace INPUT.hip and OUTPUT.so with the actual filenames. If compilation fails, read the errors, fix the source, and retry.

## Step 3: Build the submission

Create a submission .py file that:

1. Embeds the .hip source as a raw string `r"""..."""`
2. Uses `load_inline` to JIT-compile on the runner
3. Warms up the kernel with dummy data
4. Uses the kernel in `custom_kernel()` for target shapes, with preshuffle fallback for others

Template structure:
```python
#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X

import torch, os, json
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'

from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
# ... config setup ...

_hip_src = r"""
... HIP kernel source ...
"""

_cpp_src = r"""
#include <torch/extension.h>
torch::Tensor run_FUNC(torch::Tensor A, ...);
"""

_mod = load_inline(name='NAME', cpp_sources=_cpp_src, cuda_sources=_hip_src,
    functions=['run_FUNC'], verbose=False,
    extra_cuda_cflags=['-O3', '-w', '-mcumode', '--offload-arch=gfx950'])

# Warmup ...

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    # Use _mod.run_FUNC() for target shapes
    # Fall back to preshuffle for others
```

Include the full preshuffle config block from submission.py for the fallback path.

## Step 4: Submit

```bash
# Correctness first
popcorn submit --mode test --no-tui SUBMISSION.py

# Then benchmark
popcorn submit --mode benchmark --no-tui SUBMISSION.py

# Profile (shows GPU-side breakdown)
popcorn submit --mode profile --no-tui SUBMISSION.py
```

Rate limit: 6 per mode per hour. Check results for all 6 shapes.

## Reminders

- The .so from Docker cross-compilation may NOT work on the runner (ROCm version mismatch). Always use `load_inline` for the actual submission.
- Docker compilation is just a fast syntax check — the real compilation happens on the runner.
- GPU time is what matters. CPU dispatch overhead is invisible to the benchmark's CUDA event timing.
- The Triton preshuffle kernel achieves 8.5µs for K=512 M=32 shapes. Our custom kernels need to beat that.
