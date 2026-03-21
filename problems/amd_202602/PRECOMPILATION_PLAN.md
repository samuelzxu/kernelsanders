# Precompilation Plan: Cross-Compile for gfx950 Without AMD GPU

NOTE: Rate limits are implemented, currently set to 1/h for leaderboard, 6/h for benchmark/test

The submission command is:
popcorn-cli submit --leaderboard {leaderboard_name} --mode {mode} --gpu {gpu_name} {submission_file} --no-tui

Remember to validate your changes with these resources:
https://deep-reinforce.com/defense_kernel_hack.html and
https://gist.githubusercontent.com/hargup/4897f33df0d2425ac4c8c99dc8f6ec00/raw/db903b7d62bb9530596af3f147eaff535ef2e750/gpu_mode_anti_cheating_prompt

Your work should be in the spirit of engineering optimization, not a hack. Remember to avoid the keyword "stream" in the
submission file - it will cause the submission to fail.

Do not continually resubmit our best version. The leaderboard always chooses the best submission automatically.

## Key Insight

You can compile HIP kernels for gfx950 (MI355X) without having an AMD GPU,
using just the ROCm toolchain in Docker. This unblocks all custom kernel work
that was previously blocked by 12-17 minute JIT timeouts on the runner.

Reference: https://gist.github.com/gau-nernst/5f81dbb753a0afba7f69144e3af386eb

## What You Can Compile on CPU

### 1. HIP C++ kernels via `hipcc`

```bash
hipcc -O3 -g -c --offload-arch=gfx950 my_kernel.cpp
```

This cross-compiles to gfx950 device code without needing the GPU. You'd then
ship the `.o` or `.so` and load via `ctypes` or `torch.ops.load_library()`.

### 2. AITER pre-built modules

The ATOM Dockerfile already does this:

```dockerfile
PREBUILD_KERNELS=1 GPU_ARCHS="gfx942;gfx950" python3 setup.py develop
```

This builds all AITER JIT modules (including `module_moe_asm`, `module_mla_metadata`,
CK 2-stage `.so` files) at Docker build time — no GPU required. These are the same
modules that cause 30-135s JIT timeouts on the runner.

### 3. Triton AOT compilation

Triton supports ahead-of-time compilation with:

```python
target = GPUTarget("hip", "gfx950", 64)
```

You can compile Triton kernels to `.hsaco` binaries on CPU and ship them.

## Concrete Plan

1. **Build the Docker image** from gau-nernst's Dockerfile (or the ATOM Dockerfile
   with `GPU_ARCH=gfx950`)
2. **Inside the container**, compile:
   - The MXFP4 fused attention Triton kernel (#243) → `.hsaco`
   - The fmoe_g1u1 1-stage assembly module → `module_moe_asm.so`
   - Custom HIP C++ kernels for shape-specific GEMM → `.so`
3. **Ship the binaries** alongside `submission.py`, load them at runtime

## Docker Setup

### Option A: gau-nernst's lightweight image (just hipcc + PyTorch)

Dockerfile from https://gist.github.com/gau-nernst/5f81dbb753a0afba7f69144e3af386eb:

```dockerfile
FROM ghcr.io/actions/actions-runner:latest
# Installs ROCm 7.1 toolchain + PyTorch 2.10.0 rocm7.1
# Sets CXX=clang++, ROCM_PATH=/opt/rocm
# hipcc available for --offload-arch=gfx950
```

Build and use:
```bash
docker build -t rocm-cross .
docker run -it -v $(pwd):/workspace rocm-cross bash
# Inside container:
hipcc -O3 -g -c --offload-arch=gfx950 -save-temps=obj my_kernel.cpp
```

### Option B: Full ATOM image (includes AITER + Triton + everything)

```bash
cd ATOM/docker
docker build \
  --build-arg GPU_ARCH=gfx950 \
  --build-arg PREBUILD_KERNELS=1 \
  --build-arg MAX_JOBS=64 \
  -t atom-gfx950 .
docker run -it -v $(pwd):/workspace atom-gfx950 bash
```

This builds AITER with all JIT modules pre-compiled for gfx950. The resulting
`.so` files can be extracted and shipped with submissions.

## Key Consideration: Version Matching

The runner's exact ROCm version and library versions need to match. From the
ATOM Dockerfile:

- Base: `rocm/pytorch:latest` (likely ROCm 7.x)
- Triton: `release/internal/3.5.x` branch from ROCm/triton
- AITER: latest HEAD with `PREBUILD_KERNELS=1`

The gau-nernst gist uses ROCm 7.1 (`amdgpu-install_7.1.70100-1`), which should
be close enough. The important thing is that `hipcc --offload-arch=gfx950` produces
valid device binaries without the GPU present — the compiler only needs the ISA
spec, not the hardware.

## What This Unblocks

### Mixed-MLA (currently ~64µs, top is 4.3µs)

| Kernel | Previous Blocker | Expected Impact |
|--------|-----------------|-----------------|
| MXFP4 fused attention (#243) | 12-min Triton JIT timeout | 30-50% on large configs (bs=256/kv=8192) |
| Custom HIP flash-decode with MXFP4 KV | 17-min HIP compile timeout | 1.87x bandwidth savings over fp8 |
| Triton MLA decode rope (#239) | JIT timeout | Bypasses metadata overhead |

### MOE-MXFP4 (currently ~153µs, top is 115µs)

| Kernel | Previous Blocker | Expected Impact |
|--------|-----------------|-----------------|
| Triton matmul_ogs (moe_gemm_a4w4) | Triton JIT timeout (~120-200s) | 20-30% from fused routing+GEMM+SiLU |
| 1-stage fmoe_g1u1 assembly | 30s module_moe_asm JIT + cold runner timeout | 15-25% on E=257 shapes |
| fused_moe_mxfp4_silu (Triton) | JIT timeout risk | 10-20% fused GEMM+SiLU |
| Pre-built CK 2-stage modules | 105s module_moe_ck2stages JIT | Eliminates cold start entirely |

### MXFP4-MM (currently ~12µs, top is 4.3µs)

| Kernel | Previous Blocker | Expected Impact |
|--------|-----------------|-----------------|
| Shape-specific HSACO binaries | No MI355X access | 15-30% from perfect tile configs |
| Custom fused quant+GEMM HIP kernel | HIP compile timeout | Eliminates quant kernel launch |

## Anti-Cheat Compliance

Per the competition rules (https://deep-reinforce.com/defense_kernel_hack.html
and the gist review framework):

**ALLOWED:** "Emitting or selecting a specialized kernel for the current (m, n, k)
as long as it performs genuine computation for that invocation"

**ALLOWED:** Per-shape execution-path selection with precompiled kernels

**BANNED:** Persistent cross-invocation caching of outputs, harness-aware
special-casing by seed/signature, precomputed outputs

Shipping precompiled `.so`/`.hsaco` binaries is explicitly allowed as long as the
kernel performs genuine computation on the provided inputs.

## Embedding Binaries in Submission

Since submissions are single Python files, embed compiled binaries as base64:

```python
import base64, bz2, os, tempfile

# Embedded pre-compiled kernel (bz2 compressed, base64 encoded)
_KERNEL_BZ2_B64 = "QlpoOTFBWSZTW..."  # truncated

def _install_precompiled():
    data = bz2.decompress(base64.b64decode(_KERNEL_BZ2_B64))
    path = "/home/runner/aiter/aiter/jit/module_moe_ck2stages_fp4x2_fp4x2_preshuffle_on_b16_silu_per_1x32_mulWeightStage2_.so"
    if not os.path.exists(path):
        with open(path, 'wb') as f:
            f.write(data)

_install_precompiled()
```

This writes the pre-compiled .so to AITER's JIT cache BEFORE any AITER imports,
making every cold runner into a warm runner.
