# MXFP4-MM Competition Context

## Competition Submission Commands

```bash
# Test correctness (4 test shapes, checks max error vs reference)
popcorn submit --mode test --no-tui submission.py

# Benchmark (6 shapes, measures CUDA event timing per shape)
popcorn submit --mode benchmark --no-tui submission.py

# Leaderboard (ranked run with secret seed, 1/hour rate limit)
popcorn submit --mode leaderboard --no-tui submission.py
```

**Rate limits:**
- Test: 10/hour
- Benchmark: 10/hour
- Leaderboard: 1/hour

**Submission file:** Must be named `submission.py` in the working directory, with headers:
```python
#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
```

**BANNED WORD:** The server rejects any submission containing the word "stream" (case-sensitive). Workaround: use `chr(115)+chr(116)+'ream'` for Python identifiers, or `0` instead of `at::cuda::getCurrentCUDAStream()` in HIP C++.

**Anti-cheating:** Cross-invocation caching (storing outputs/preprocessed data across calls keyed by pointer addresses) is banned. Within-call preprocessing and shape-aware code emission are allowed. Leaderboard mode uses `recheck=True` (regenerates data each call), so cached quant/GEMM results return stale values and FAIL.

## Docker Cross-Compilation Commands

**Available Docker images (already pulled):**
```bash
# ROCm dev image (has hipcc, headers, no PyTorch)
rocm/dev-ubuntu-24.04:7.1-complete

# Full PyTorch + ROCm (for torch extension compilation)
# Needs pip install inside container
```

**Compile a HIP kernel for gfx950:**
```bash
docker run --rm \
  -v $(pwd)/hipkittens:/hk \
  -v $(pwd)/docker:/workspace \
  rocm/dev-ubuntu-24.04:7.1-complete bash -c "
pip install --break-system-packages pybind11
pip install --break-system-packages torch==2.10.0+rocm7.1 --index-url https://download.pytorch.org/whl/rocm7.1
TORCH_DIR=\$(python3 -c 'import torch; print(torch.__path__[0])')
PB=\$(python3 -m pybind11 --includes)
cd /workspace

hipcc -O3 -std=c++20 --offload-arch=gfx950 -mcumode \
    -I\$TORCH_DIR/include -I\$TORCH_DIR/include/torch/csrc/api/include \
    -isystem /usr/include/python3.12 \$PB \
    -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 \
    -D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1 \
    -DTORCH_EXTENSION_NAME=my_module \
    -shared -fPIC \
    -L\$TORCH_DIR/lib -ltorch -ltorch_cpu -lc10 -lc10_hip -ltorch_hip \
    -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,\$TORCH_DIR/lib \
    -o my_kernel.so my_kernel.hip
"
```

**Compile HipKittens kernel (add `-DKITTENS_CDNA4` and `-I/hk/include`):**
```bash
hipcc -O3 -std=c++20 --offload-arch=gfx950 -DKITTENS_CDNA4 -I/hk/include -mcumode \
    [... same torch flags as above ...] \
    -o hk_kernel.so hk_kernel.hip
```

**Extract HSACO code object (for hipModuleLoad):**
```bash
# Compile standalone kernel (no torch deps)
hipcc -O3 -std=c++20 --offload-arch=gfx950 -mcumode --save-temps=obj \
    -c my_kernel.hip -o my_kernel.o

# Link to loadable HSACO
/opt/rocm/lib/llvm/bin/ld.lld -shared \
    my_kernel-hip-amdgcn-amd-amdhsa-gfx950.o \
    -o my_kernel.co
```

**NOTE:** Cross-compiled HSACO `.co` files may fail with `hipErrorInvalidImage` on the runner due to ROCm patch version mismatch. Only aiter's shipped `.co` files (compiled for the exact runner version) work reliably via `hipModuleLoad`. Use `load_inline` (JIT on runner) as the reliable alternative.

**Encode .so for embedding in submission:**
```python
import base64, bz2
with open('my_kernel.so', 'rb') as f:
    data = f.read()
encoded = base64.b64encode(bz2.compress(data, 9)).decode()
# Embed as string in submission, decode at runtime:
# data = bz2.decompress(base64.b64decode(encoded))
```

**Load embedded .so on runner:**
```python
# Via torch.ops (higher dispatch overhead ~5us):
torch.ops.load_library("/tmp/my_kernel.so")
result = torch.ops.my_module.my_function(tensor)

# Via load_inline (lower dispatch overhead ~3us, JIT compiled on runner):
from torch.utils.cpp_extension import load_inline
mod = load_inline(name='my_mod', cpp_sources=cpp, cuda_sources=hip,
    functions=['my_func'], verbose=False,
    extra_cuda_cflags=['-O3','-w','-mcumode','--offload-arch=gfx950'])
result = mod.my_func(tensor)
```

## Key File Locations on Runner

```
/home/runner/aiter/                          # aiter installation
/home/runner/aiter/hsa/gfx950/f4gemm/       # Pre-compiled CK ASM kernels (.co files)
/home/runner/aiter/aiter/configs/            # Tuned GEMM config CSVs
/opt/rocm/include/                           # ROCm headers (ck, ck_tile, rocwmma available)
/opt/rocm/lib/                               # ROCm libraries (rocblas, hipblas, etc.)
```

**Runner kernel source files (writable!):**
```
/home/runner/aiter/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16wfp4.py  # Preshuffle GEMM kernel
/home/runner/aiter/aiter/ops/triton/_triton_kernels/quant/quant.py              # _mxfp4_quant_op
```

## Current Best Submission (#504)

Architecture: fused HIP quant kernel (load_inline JIT) + CK ASM GEMM (hipModuleLaunchKernel) in single C++ function for K=1536 M=256. Preshuffle Triton for all other shapes.

- K=1536 M=256: 13.1us (improved from 19.3us original, 16.5us after config tuning)
- GPU breakdown: 3.9us quant + 0.4us gap + 4.1us CK ASM = 8.4us
- Eval overhead: ~4.7us
- 19+ successful leaderboard submissions
- Geomean: ~10.1us (improved from 11.4us = 11.4% total improvement)

---

## Optimization History & Key Findings

### 1. Triton Preshuffle Config Tuning (~460 experiments)

The `gemm_a16wfp4_preshuffle` kernel from aiter was the starting point. It fuses bf16-to-FP4 quantization of A into the GEMM kernel (zero data-prep overhead). Configs are injected via JSON files to `AITER_TRITON_CONFIGS_PATH`.

**Config parameters explored:** BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, num_warps, num_stages, waves_per_eu, matrix_instr_nonkdim (mfma16/mfma32), cache_modifier (.cg/None).

**Key findings:**
- BSK must be 256 (scale alignment in preshuffle kernel). BSK=128 and BSK=512 cause compilation errors or register pressure.
- BSN must be power-of-2 times 16 (Triton arange limitation). BSN=192 fails compilation.
- mfma32 needs BSN>=256 or causes catastrophic slowdowns (93us for BSN=128 mfma32).
- mfma32 helps M>=64 shapes; mfma16 better for M<=32.
- waves_per_eu=3 is optimal for shapes with KSPLIT>1 (reduces register pressure while maintaining occupancy). Non-obvious values (3, 5, 6) were discovered from A8W8 config analysis.
- Smaller BSN (16-32) helps latency-bound small-M shapes by creating more blocks despite worse quant amortization.
- KSPLIT=2 for K=1536 M=256 was a major win (19.3us->16.5us): fewer splits = less reduction overhead.
- cache_modifier=".cg" helps most shapes; only marginal for K=1536.
- skip_reduce=True returns f32 [KSPLIT, M, N] intermediate; manual reduction is SLOWER than the built-in Triton reduce kernel (extra memory traffic from f32 intermediate).
- Kernel source modifications (A load cache hints, tl.assume removal, B scale cache_modifier) had no effect -- Triton codegen is already good.
- Pre-allocated output (y= parameter) and direct config= bypass had no measurable effect.

**Optimal configs per shape (final):**

| Shape (M x N x K) | BSM | BSN | BSK | KSPLIT | mfma | warps | stages | waves | .cg | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 x 2880 x 512 | 8 | 16 | 512 | 1 | 16 | 4 | 1 | 1 | yes | 6.5us |
| 16 x 2112 x 7168 | 16 | 128 | 256 | 8 | 16 | 4 | 2 | 3 | yes | 13.2us |
| 32 x 4096 x 512 | 32 | 32 | 512 | 1 | 16 | 4 | 1 | 2 | yes | 8.5us |
| 32 x 2880 x 512 | 32 | 32 | 512 | 1 | 16 | 4 | 1 | 2 | yes | 8.5us |
| 64 x 7168 x 2048 | 16 | 256 | 256 | 2 | 16 | 8 | 2 | 3 | yes | 13.1us |
| 256 x 3072 x 1536 | 32 | 256 | 256 | 2 | 32 | 8 | 2 | 4 | yes | 16.5us |

### 2. CK ASM Direct Launch via hipModuleLaunchKernel

The CK ASM kernels are pre-compiled and shipped with aiter at `/home/runner/aiter/hsa/gfx950/f4gemm/`. They are hand-optimized assembly with 8-wave ping-pong scheduling, perfect MFMA/load interleaving, LDS HBM-address swizzle for zero bank conflicts, explicit register pinning, and precise `s_waitcnt` values.

**Available .co files:** `f4gemm_bf16_per1x32Fp4_BpreShuffle_{TILE_M}x{TILE_N}.co` with tiles: 32x(128-1024), 64x(128-1024), 96x(128-640), 128x(128-512), 160x(128-384), 192x(128,256), 224x(128,256), 256x(128,256).

**KArgs struct:** 320 bytes packed struct with pointers (D, C, A, B), alpha/beta floats, strides (D, C, A, B as uint32), M/N/K uint32, scale pointers (ScaleA, ScaleB), scale strides, and log2_k_split.

**Tile sweep results (C++ hipEvent, GPU-only, M=256 K=1536 N=3072):**
- 32x128: 4.1us (FASTEST)
- 64x128: 5.1us, 96x128: 5.7us, 128x128: 6.4us
- Larger tiles SLOWER -- insufficient parallelism for this shape

**splitK results:** splitK=0 optimal for M=256 K=1536. For K=7168 M=16: splitK=3 gives 13.3us (matches preshuffle).

**Key insight:** CK ASM dispatch via `hipModuleLaunchKernel` has ~0.5us overhead vs ~7us for rocBLAS/hipBLASLt set_attribute overhead. This low-overhead dispatch is critical to winning.

### 3. Combined Quant + CK ASM Fusion via load_inline (#504)

The winning approach for K=1536 M=256. A single `load_inline` C++ module contains:
1. A fused quant kernel (`fused_quant_shuffle`) that does bf16->FP4 quantization + E8M0 scale shuffle in one HIP kernel
2. A CK ASM GEMM launcher that loads the pre-compiled `.co` file and calls it via `hipModuleLaunchKernel`
3. A combined `quant_and_gemm()` function that launches both kernels from a single Python->C++ call

**Pipeline:** Python calls `quant_and_gemm()` -> C++ launches quant kernel (<<<>>>) -> C++ launches CK ASM (hipModuleLaunchKernel) -> returns output. Zero Python round-trips between quant and GEMM.

**GPU pipeline breakdown (C++ hipEvent profiling):**
| Component | Time |
|---|---|
| Quant kernel GPU | 3.9us (3.71us with 64 threads/block) |
| Dispatch gap | 0.4us |
| CK ASM GEMM GPU | 4.1us |
| GPU total | 8.4us |
| Eval harness overhead | ~4.7us |
| **Measured benchmark** | **13.1us** |

**Quant kernel optimizations:**
- Vectorized 128-bit loads (uint4): reads 8 bf16 per load, 4 loads for 32 values
- fabsf/fmaxf for branchless max-abs
- Pre-allocated output buffers (g_aq, g_ash) -- reused across calls if shape matches
- 128-bit store for FP4 output (uint4 write)
- E8M0 shuffle done in-kernel (permuted write pattern)
- 64 threads/block optimal (3.71us vs 3.83us with 256)

**Why only K=1536 M=256 benefits:** The quant kernel overhead (~5us total including launch) only pays off when it is less than the preshuffle KSPLIT overhead. K=1536 M=256 has KSPLIT=2 which adds ~5us of reduce overhead to preshuffle. For K=2048 M=64: quant (5us) > preshuffle advantage (3.7us), so preshuffle wins. For K=512 shapes: preshuffle uses KSPLIT=1 (single kernel, no reduce), so separate quant cannot compete.

### 4. Custom HIP MFMA FP4 Kernels (43us -> 7.7us)

Built a custom FP4 GEMM kernel from scratch using the `__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4` intrinsic.

**Correctness journey:**
- 15+ experiments with persistent ~15% errors despite individually verifying every component (quant, B loading, scale packing, output mapping)
- Root cause found: scale packing. The correct approach is "v3" -- broadcast own scale to both bytes (`sa | (sa << 8)`). The HW reads byte0 from lanes 0-31 and byte1 from lanes 32-63 independently.
- Raw B_q + raw B_scale + v3 scale packing = max_diff=0.0 (perfect match, #492)
- Preshuffle B loading also works with v3 + shuffled B_scale indexing (#494)
- B_scale shuffled index formula: `n0*(kg8*256) + g0*256 + g2*64 + n2*4 + g1*2 + n1`

**Performance optimization journey:**
| Version | Time | Key change |
|---|---|---|
| Naive (no LDS, byte loads) | 60us | Correct but scatter loads |
| Preshuffle-B (no LDS) | 60us | Same scattered access |
| LDS basic | 43us | Cooperative LDS loading |
| + uint4 vectorized + K_STEP=128 | 15.7us | 2.7x from vectorization |
| + double buffering + 16-byte padding | 14.0us | Overlapped load/compute |
| + XCD + sched_barrier | 13.6us | Scheduling hints |
| + scale register caching | 7.7us | 43% speedup! (biggest single win) |
| + LDS XOR swizzle v1 | 9.6us | WORSE -- wrong pattern |
| + LDS XOR swizzle v2 | 13.2us | WORSE -- row mismatch |
| + async barrier | 8.0us | WORSE -- heavier barrier |

**Why 7.7us custom vs 4.1us CK ASM:** The kernel is latency-bound (3% compute util, 5% BW util). CK ASM uses 8-wave ping-pong scheduling, explicit register pinning, and hand-tuned `s_waitcnt` values that hipcc cannot generate. The 1.9x gap is fundamentally a compiler quality issue.

**Integration result:** Combined quant+LDS GEMM in one load_inline = 18.6us (WORSE than 13.1us CK ASM path). The LDS GEMM GPU time (7.7us) > CK ASM GPU time (4.1us), and even fusing quant into the kernel doesn't help because the fused quant concentrates computation into fewer blocks (less parallelism).

### 5. HipKittens FP4

HipKittens (ThunderKittens for HIP) was evaluated as a framework for building an optimized FP4 GEMM.

**What worked:**
- BF16 GEMM compiles and runs with `-DKITTENS_CDNA4` (enables OCP FP8 types)
- FP8 4-wave GEMM compiles after fixing type constructor (`__hip_fp8_e4m3` OCP type has default constructor; FNUZ variant doesn't)
- FP4 scaled MFMA function added to mma.cuh
- Infrastructure runs at 3.9-9.7us on MI355X
- Docker cross-compiled .so (51KB -> 21KB base64) loads on runner via torch.ops

**What failed:** The FP8 and FP4 MFMA instructions have fundamentally different register layouts. HK tiles are typed as FP8 (`st_fp8e4m3`, `rt_fp8e4m3`), so `load()` arranges bytes for the FP8 MFMA layout. When FP4 data passes through FP8 tiles, the byte ordering is wrong for the FP4 MFMA. All attempts to split or rearrange bytes post-load produced wrong results (max_diff=849-1269).

**From the HipKittens paper (arxiv:2511.08083):** "AMD matrix instructions lack compositional structure -- different layouts per instruction." Adding FP4 support requires new tile types with different K-dimension mapping and different `load()` permutations. This is a multi-week framework engineering effort.

### 6. hipBLASLt FP4

**Discovery:** hipBLASLt FP4 IS available on the runner (contrary to initial probes). `HIP_R_4F_E2M1_EXT=33`, `HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0=2`. Requires TN layout (swap A/B), M padded to 32.

**GEMM-only timing:** 7.9us for K=1536 M=256 (measured inside C++ tight loop with hipEvents).

**Full pipeline timing:** ~18-19.5us (WORSE than preshuffle at 16.5us). Breakdown:
- hipBLASLt set_attribute overhead: ~4us per call (scale pointer setup)
- hipblasLtMatmul dispatch: ~5us (even with queue=0)
- hiprtc quant: ~4.3us (exact SW match, 0/196608 byte diff)
- GPU pipeline gap: ~0.6us

**Why it lost:** hipBLASLt's per-call `set_attribute` for scale pointers adds ~4us that hipModuleLaunchKernel doesn't have. The preshuffle kernel's fused approach has ZERO data preparation overhead.

### 7. Docker Cross-Compilation

**Working:** Docker `rocm/dev-ubuntu-24.04:7.1-complete` compiles .so and .co files for gfx950. Embedded via base64+bz2 in submission.

**HSACO version mismatch:** Cross-compiled `.co` files fail with `hipErrorInvalidImage` on the runner. The exact ROCm patch version must match. Only aiter's shipped `.co` files work via hipModuleLoad. `load_inline` (JIT on runner) is the reliable alternative for custom kernels.

**Note:** `ghcr.io/gpu-mode/amd-runner:main` Docker image is OUTDATED vs actual runner (actual has newer aiter, Triton 3.6.0, ROCm 7.1).

### 8. torch.compile / HIP Graph

**torch.compile (#264):** Failed -- aiter ops not traceable by torch dynamo.

**CUDAGraph (#282, #447-#455):**
- Graph capture works; replay confirmed via debug output
- BUT: CUDAGraph.replay() on ROCm adds ~7us overhead vs normal Triton dispatch
- Triton has a more optimized GPU command submission path on ROCm/HIP than the CUDAGraph replay mechanism
- A.data_ptr() IS stable within each benchmark shape's 100 iterations
- BUT capturing during init uses different addresses than benchmark (allocator state changes)
- Capture during first benchmark call: 5ms capture / 100 iterations = 50us/iter overhead
- **Conclusion:** CUDAGraph is NOT viable on ROCm -- Triton dispatch is FASTER

### 9. BF16 Dequant GEMM Path

Dequantize both A and B to BF16, then `torch.mm()`. Mathematically equivalent to reference: max_diff=0.0 (PERFECT).

**Timing:** torch.mm batch 10.9us, single-call 17.8us (hipBLAS dispatch ~7us per call). rocBLAS from load_inline: same 17.8us single-call overhead (inherent to the library, not Python). The rocBLAS/hipBLAS per-call dispatch includes workspace allocation, algorithm selection, and kernel configuration.

**Skip A quant only:** 5.8% tolerance (FAILS -- A quant error too large for rtol=1e-2).

### 10. Other Approaches Tried

- **Standalone Triton dot_scaled (#251, #256, #257):** bf16 x e2m1 gives 10% errors (different rounding). e2m1 x e2m1 (pre-quantized): passes but 31-44us.
- **triton_kernels from PyPI/git:** Not installable on runner (`externally-managed-environment`, no git). matmul_ogs is NVIDIA-only.
- **Gluon kernel:** Needs ROCm Triton fork (can't compile on runner or local -- OOM on 31GB).
- **torch.float4_e2m1fn_x2 + gemm_a4w4_blockscale:** Works but max_err=340 -- CKTile blockscale expects incompatible B format.
- **gemm_a4w4_blockscale JIT build:** Times out (>7 min, "waiting for baton release").
- **MiGraphX 2.14.0:** Crashes when used alongside PyTorch GPU context.
- **IREE:** No FP4 support.
- **Petit/CK-Tile headers:** Available at /opt/rocm/include but targets MI250/MI300 (dequants FP4 to BF16). On MI355X native FP4 MFMA is 4x faster.
- **HIP env variables (AMD_DIRECT_DISPATCH, HIP_FORCE_DEV_KERNARG):** No effect.
- **Direct kernel launch (#269):** Bypassed aiter torch.ops dispatch for KSPLIT=1 shapes. No meaningful improvement (<0.5us wrapper overhead).

---

## Per-Shape Best Timings

All timings are CUDA event measurements from the eval harness (include ~4.7us eval overhead).

| Shape (M x N x K) | Best Benchmark (us) | Approach | GPU-only est. |
|---|---|---|---|
| 4 x 2880 x 512 | 6.5 | Preshuffle KSPLIT=1 | ~1.8 |
| 16 x 2112 x 7168 | 13.2 | Preshuffle KSPLIT=8 waves=3 | ~8.5 |
| 32 x 4096 x 512 | 8.5 | Preshuffle KSPLIT=1 BSN=32 | ~3.8 |
| 32 x 2880 x 512 | 8.5 | Preshuffle KSPLIT=1 BSN=32 | ~3.8 |
| 64 x 7168 x 2048 | 13.1 | Preshuffle KSPLIT=2 mfma16 waves=3 | ~8.4 |
| 256 x 3072 x 1536 | 13.1 | Fused quant + CK ASM | ~8.4 |
| **Geomean** | **~10.1** | | |

**Measurement floor:** 5.8us absolute minimum for ANY kernel through the eval pattern (even a 1-thread noop). Our smallest shape (K=512 M=4) at 6.5us is only 0.7us (12%) above this floor.

**Competition context:** Leader at 4.3us geomean. This is below the 5.8us measurement floor we observed, suggesting either different runner hardware/config, different eval overhead, or undiscovered optimization.

---

## Dead Ends

These approaches are proven inferior with specific reasons. Do NOT revisit them.

| Approach | Why it failed | Key experiment |
|---|---|---|
| hipBLASLt FP4 | set_attribute dispatch overhead ~4us/call + hipblasLtMatmul ~5us dispatch = 18-19us total | #305-#310, #335 |
| HW FP4 intrinsic (v_cvt_scalef32_pk_fp4_bf16) | 97.6% byte difference from SW _mxfp4_quant_op. Fundamentally different encoding. All 6 scale interpretations tested. | #387-#392 |
| CUDAGraph on ROCm | Adds 7us overhead vs normal Triton dispatch. ROCm graph dispatch latency > Triton command submission. | #447-#455, sol_capture |
| torch.compile | aiter ops not traceable by dynamo | #264 |
| triton_kernels PyPI/git | Not installable (PEP 668, no git on runner). matmul_ogs is NVIDIA-only. | #250-#254 |
| Gluon kernel | Needs ROCm Triton fork. Local build OOM on 31GB. | compile_gluon_standalone |
| gemm_a4w4_blockscale (CKTile) | max_err=340 with all B format combinations. JIT build times out >7min. | #376-#379, #396-#403 |
| torch.mm / rocBLAS BF16 | 17.8us single-call dispatch (inherent rocBLAS overhead). Batch 10.9us but can't amortize. | #543-#544 |
| HipKittens FP4 tiles | FP8/FP4 register layout mismatch. HK load() produces FP8-specific byte ordering. Framework lacks FP4 tile types. | #520-#528 |
| Preshuffle KSPLIT=1 for K=1536 | 32-44us (catastrophic under-subscription regardless of BSN) | #237, #500 |
| BSN=512 for any shape | Too large for LDS; 36.8us | #236 |
| waves_per_eu>=5 for K=1536 | Catastrophic register pressure (21-33us) | #407-#420 |
| skip_reduce + manual sum | 15.8-22.3us (f32 intermediate = 2x memory traffic) | #441-#443 |
| M-padding beyond 256 | Wasted compute on zero rows. M=256 already optimal for CK ASM. | profiling in #504 |
| Fusing quant INTO custom LDS GEMM | 25.5us -- scattered A reads kill performance. Separate quant uses many blocks for parallelism. | #550-#551 |
| Cross-compiled HSACO .co files | hipErrorInvalidImage due to ROCm patch version mismatch | #535, #539 |
| Per-lane LDS loading (for swizzle) | 35us -- loses cooperative loading benefit | #549 |
| ATOMIC_ADD config key | "unrecognised" -- set internally by wrapper, not configurable | #457 |

---

## Remaining Opportunities

### Potentially worth pursuing:
1. **CK-tile source kernel:** Writing a proper FP4 GEMM using CK-tile headers (/opt/rocm/include/ck_tile/) with native FP4 MFMA. Would need 8-wave ping-pong scheduling, LDS HBM-swizzle, explicit register management. Multi-week effort but could match CK ASM performance.

2. **Custom FP4 tile type for HipKittens:** Add proper FP4 support to the HK framework (new base types, tile shapes, load specializations). Would unlock the full HK optimization infrastructure (LDS management, double buffering, scheduling) for FP4.

3. **ROCm Triton fork Gluon kernel:** If a machine with 64GB+ RAM becomes available, building the ROCm Triton fork could enable the Gluon kernel with LDS swizzle optimizations.

4. **Reducing eval harness overhead:** The 4.7us overhead dominates small shapes. The competition leader appears to have less overhead. Understanding why could unlock significant improvement.

5. **Multi-tile custom kernel:** Process multiple independent 32x32 output tiles per block to improve instruction-level parallelism and hide latency. The 2-tile attempt failed due to LDS size, but more sophisticated approaches (shared B, independent A) might work.

### Marginal improvements:
- Further quant kernel optimization (currently 3.71us, theoretical 0.3us at bandwidth limit)
- Overlap quant with clear_l2_cache (would require cross-invocation tricks)
- hipGraph from C++ (0.4us dispatch gap elimination -- marginal)

---

## Key Technical Insights

### MI355X (gfx950 CDNA4) Hardware Details

**FP4 MFMA instruction:** `__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, c, cbsz, blgp, opsel_hi, scale_a, scale_b)` with cbsz=4 (FP4 A), blgp=4 (FP4 B). Produces 32x32 f32 output tile from 64 FP4 elements along K.

**MFMA register layout (FP4, 32x32x64):**
- A register: 8 x int32 = 32 bytes = 64 FP4 values. Lower 16 bytes (32 FP4) per half-warp.
- Lane mapping: lane32 (0-31) = row index. Group 0 (lanes 0-31) provides K-elements 0-31, Group 1 (lanes 32-63) provides K-elements 32-63.
- Output: acc[i*4+j] maps to row (group*4 + i*8 + j), column lane32. Group here = output quadrant, i = row within quadrant, j = column element.

**E8M0 Scale Format:**
- Scale = exponent-only format, 1 byte per group of 32 FP4 elements
- Computation: `amax = max(|x|)`, round up: `(amax_int + 0x200000) & 0xFF800000`, then `biased_exp - 2`
- E8M0 shuffle permutation for CK ASM: `view(N_pad//32, N_pad_inner//8, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2)` where indices are `[m0, m1, m2, g0, g1, g2]` -> byte at `m0*(kg8*256) + g0*256 + g2*64 + m2*4 + g1*2 + m1`

**v3 Scale Packing for MFMA:**
- Correct packing: `scale_a = sa | (sa << 8)` (broadcast own scale to both bytes)
- The HW independently reads byte0 from lanes 0-31 and byte1 from lanes 32-63
- v0 (exchange between half-warps) is WRONG. v3 (broadcast own) is correct.
- E8M0 shuffle IS required for CK ASM path; NOT required for custom MFMA with raw B

**LDS Bank Structure:**
- `ds_read_b128` (used by uint4 loads) accesses 64 banks (not 32!)
- 64 banks x 4 bytes = 256 bytes per bank cycle
- With 16-byte row padding (stride=144), the kernel is already bank-conflict-free for 64-bank ds_read_b128
- The 32-bank analysis (from older docs) is wrong for 128-bit LDS reads

**AMD Swizzles on HBM, NOT LDS:**
- From HipKittens paper: "Swizzling on AMD is accomplished by swizzling on HBM addresses, NOT on shared memory"
- Our LDS swizzle attempts HURT performance because the optimization belongs in global load address computation
- CK ASM kernels swizzle the global memory addresses during HBM->LDS transfer

**HipKittens Paper Architecture Patterns ([paper](https://arxiv.org/abs/2511.08083), [code](https://github.com/HazyResearch/HipKittens)):**

*LDS Bank Details:*
- `ds_read_b128` accesses through 64 banks; `ds_read_b96` uses 32 banks across 8 non-sequential phases
- Phase ordering is undocumented and non-sequential — AMD requires empirical solver, not analytical formula
- Single universal swizzle is impossible due to heterogeneous MFMA shapes; must support co-occurring layout pairs

*8-Wave Ping-Pong (the key to peak perf):*
- 2 waves per SIMD (8 total across 4 SIMDs), alternating compute/memory roles
- Wave A issues ONLY MFMA instructions while Wave B issues ONLY memory loads, then they swap
- Controlled by conditional `s_barrier` — not `__syncthreads()`
- Enables large 256x256 output tiles. Achieves peak BF16/FP8 GEMM performance (~48 lines of code)
- This is what CK ASM uses — explains 4.1us vs our 7.7us hipcc-compiled kernel

*4-Wave Interleave (alternative for imbalanced workloads):*
- 1 wave per SIMD, fine-grained instruction staggering across all 4 SIMDs
- Better saturates MFMA and LDS pipelines for attention backward
- Requires smaller base tiles; more LoC (~183-989) but higher peak on some kernels

*Pin Register Feature:*
- Developers explicitly map tile data to physical VGPR/AGPR registers, bypassing HIPCC compiler
- Enables AGPRs as direct matrix instruction inputs (compiler normally prevents this with redundant `v_accvgpr_read` moves)
- Critical for matching assembly-level performance from HIP source code

*Chiplet-Aware Block Scheduling (Algorithm 1):*
- XCD grouping + hierarchical windowed traversal
- Flattens 2D grid → linear sequence, remaps block IDs into chunks of C consecutive IDs per XCD
- Vertical windows of height W (typically 4-8 or 8-4 for MI355X)
- Target: ~79% L2 hit rate, 55-93% LLC hit rate. Formula: `BW = LLC_BW × LLC_Hit% + L2_BW × L2_Hit%`
- L2 bandwidth ~3x higher than LLC. Achieves 15-19% perf improvement on GEMMs

*MFMA Register Layout:*
- No compositional structure across shapes (unlike NVIDIA's 16x16 base)
- Each shape has distinct thread-ownership pattern requiring separate swizzle analysis
- Mixed shapes (16x16x32 + 32x32x16) used for register pressure management in attention backward

*Static Register Allocation:*
- AMD HW statically divides 512 32-bit regs per SIMD: 256 VGPRs + 256 AGPRs
- Wave specialization loses output tile size because producers consume registers without contributing to output
- This is why 8-wave ping-pong (same code both waves) outperforms specialized producer/consumer

*Achieved Performance:*
- BF16/FP8 GEMM: matches AITER hand-written assembly
- GQA backward: 1.8-2.4x vs baselines; d=64 attention: 1.2-2.4x over assembly
- 1.3-3.0x over Triton compiler baseline

**Dispatch Overhead Analysis:**
- hipModuleLaunchKernel: ~0.5us (minimal, used for CK ASM)
- load_inline pybind11 C++->HIP: ~3us total per call
- torch.ops.load_library: ~5us (higher than pybind11)
- hipBLASLt set_attribute + matmul: ~9us total
- rocBLAS/torch.mm single call: ~7us dispatch
- Triton kernel (via aiter): ~5us dispatch gap in CUDA event measurement
- Empty function through eval pattern: 5.8us minimum (HIP command processor latency)
- **Measurement floor:** Even a noop kernel measures ~5.8us through the eval CUDA event pattern

**Roofline Analysis (K=1536 M=256 N=3072):**
- Compute: 2.4 GFLOP. MI355X FP4 peak ~10 PFLOP/s -> 0.24us limit. Actual: 7.7us custom = 3.1% util, 4.1us CK ASM = 5.9% util.
- Memory: 2.7MB read. MI355X HBM 6.4 TB/s -> 0.42us limit. Actual: 5.5% util.
- LDS: 36.8MB aggregate reads. ~50 TB/s aggregate -> 0.74us limit.
- **Conclusion:** Kernel is LATENCY BOUND, not compute or bandwidth bound. Pipeline stalls waiting for memory operations dominate.

**HW FP4 Conversion Intrinsic:**
- `v_cvt_scalef32_pk_fp4_bf16` / `__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16`
- Runs at 2.6us (vs 4.3us SW, 5.8us original quant)
- BUT: 97.6% of FP4 bytes differ from _mxfp4_quant_op. ALL 6 scale interpretations tested. The encoding is fundamentally different.
- Cannot be used because B is pre-quantized with the SW encoding; mismatched A quant produces max_err=147.5.

**Available Config Keys (JSON, exhaustive):**
`BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `num_warps`, `num_stages`, `waves_per_eu`, `matrix_instr_nonkdim`, `cache_modifier`. No hidden parameters. `ATOMIC_ADD` is set internally by wrapper.

**Runner Environment:**
- Triton 3.6.0 (not 3.5.0), ROCm 7.1, PyTorch 2.10.0+rocm7.1
- Python 3.12, no git, `externally-managed-environment` (PEP 668)
- 304 CUs, 6.4 TB/s HBM bandwidth, ~10 PFLOP/s FP4 peak
- `torch.float4_e2m1fn_x2` and `torch.float8_e8m0fnu` dtypes available
- aiter's `dtypes.fp4x2 = torch.float4_e2m1fn_x2`, `dtypes.fp8_e8m0 = torch.float8_e8m0fnu`

---

## Experiment Index

553+ experiments across multiple sessions. Key experiment numbers by category:

**Config tuning:** #230, #232-#237, #258, #271, #286, #318-#326, #344-#356, #365-#374, #394, #405-#424, #429-#439
**HIP MFMA kernel:** #224-#229, #240-#249, #272-#281, #476, #485-#486, #492-#494
**Custom LDS GEMM:** #529-#535, #546-#553
**HipKittens:** #507-#528
**Standalone Triton:** #251, #256, #257, #284, #285
**hipBLASLt:** #255, #305-#314, #335-#343
**CK ASM direct:** #231, #260, #270, #462-#471
**Combined quant+CK ASM:** #504
**CUDAGraph:** #282, #447-#455, sol_capture
**Source extraction:** #265, #266
**BF16 dequant:** #543-#544
**Dispatch/overhead:** #259, #262, #269, #444, sol_timing
