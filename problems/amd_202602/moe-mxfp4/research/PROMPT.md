# MOE-MXFP4 Kernel Optimization — Continuation Prompt

## Problem
Optimize a Mixture-of-Experts (MOE) kernel with MXFP4 (microscale FP4) quantized weights on AMD MI355X (gfx950) GPUs for the GPU MODE competition. The leaderboard ranks by geometric mean of execution times across 7 benchmark shapes.

## Current Standing
- **Our best score**: ~151µs geomean, **rank #11** out of 70 participants
- **Top score**: 114.6µs (#1, John Hahn)
- **#3**: 141.3µs
- **submission.py** = `103_v98_opus_only.py` (safe baseline, always works)
- **Total experiments so far**: 142+

## Competition Rules
- Submit via: `popcorn-cli submit --leaderboard amd-moe-mxfp4 --mode {test|benchmark|leaderboard} --gpu MI355X submission.py --no-tui 2>&1`
- Rate limits: 1/hr leaderboard, 6/hr test/benchmark
- Anti-cheating: https://gist.githubusercontent.com/hargup/4897f33df0d2425ac4c8c99dc8f6ec00/raw/db903b7d62bb9530596af3f147eaff535ef2e750/gpu_mode_anti_cheating_prompt
- **ALLOWED**: Per-shape kernel specialization, AOT-compiled kernels, pre-compiled binaries — as long as they perform genuine computation
- **BANNED**: Cross-invocation caches of outputs, precomputed replay, harness-aware shortcuts
- **CRITICAL**: Avoid the word "stream" in submission files (causes rejection)
- Do NOT repeatedly resubmit the same version — leaderboard keeps best score automatically

## Benchmark Shapes (7 total)
```
E=257 bs=16:   d_expert=256, d_hidden=7168, topk=9  (very sparse, tok/exp=0.56)
E=257 bs=128:  d_expert=256, d_hidden=7168, topk=9  (sparse, tok/exp=4.48)
E=257 bs=512:  d_expert=256, d_hidden=7168, topk=9  (moderate, tok/exp=17.9)
E=33  bs=16:   d_expert=512, d_hidden=7168, topk=9  (sparse, tok/exp=3.9)
E=33  bs=128:  d_expert=512, d_hidden=7168, topk=9  (moderate, tok/exp=34.9)
E=33  bs=512:  d_expert=512, d_hidden=7168, topk=9  (dense, tok/exp=139.6)
E=33  bs=512:  d_expert=2048, d_hidden=7168, topk=9 (dense, tok/exp=139.6)
```

Test shapes (must also pass correctness):
```
test1: bs=8,   d_expert=1024, E=257, topk=9
test2: bs=32,  d_expert=2048, E=33,  topk=9
test3: bs=128, d_expert=1536, E=65,  topk=7
```

## Current v103 Architecture (what works)
The baseline uses AITER's CK (Composable Kernel) 2-stage pipeline:
```
sort tokens → quant hidden_states → CK_stage1_GEMM+SiLU → requant → CK_stage2_GEMM
(5 kernel launches, ~152µs geomean)
```

Key optimizations in v103:
1. **cktile BF16 path** for sparse shapes (tok/exp < 5: split_k=2, tok/exp < 40 + E≤33: split_k=1) — skips quant entirely, 30-40% faster per shape
2. **Pre-allocated sorting buffers** — avoids repeated allocation
3. **Opus sorting** (`moe_sorting_opus_fwd`) — marginally faster than standard
4. **`@torch.inference_mode()`** — eliminates autograd overhead

Per-shape timings (v103):
```
E=257 bs=16:   ~92µs  (cktile sk=2)
E=257 bs=128:  ~180µs (cktile sk=2)
E=257 bs=512:  ~255µs (default CK FP4)
E=33  bs=16:   ~61µs  (cktile sk=2)
E=33  bs=128:  ~102µs (cktile sk=1)
E=33  bs=512 d=512:  ~216µs (default CK FP4)
E=33  bs=512 d=2048: ~353µs (default CK FP4)
```

## The Two Breakthrough Paths Forward

### Path 1: Pre-compiled CK Modules (Eliminates Cold Runner Timeouts)

**Problem**: Cold runners spend ~310s on JIT compilation before any computation:
```
module_moe_sorting_opus:     27s
module_moe_cktile2stages:   111s  ← CK templates
module_activation:            23s
module_moe_sorting:           23s
module_moe_ck2stages_fp4x2: 105s  ← CK templates
module_quant:                 25s
Total JIT:                  ~314s of 720s budget
```

**Solution**: Pre-compile these `.so` files via Docker cross-compilation for gfx950, upload to GitHub releases, download at runtime. v142 proved this works — the runner successfully loaded a pre-downloaded `module_moe_sorting_opus.so` (saved 27s).

**Status**:
- GitHub repo created: `samuelzxu/aiter-precompiled`
- v0.1-rocm72 release has 3 modules (sorting, quant, enum) built with ROCm 7.2 — ABI compatible for sorting/quant
- **BLOCKER**: CK modules (`module_moe_ck2stages`, `module_moe_cktile2stages`) failed to compile due to ROCm version mismatch (we had 7.2, runner has 7.1)
- Docker build on ARM Mac with Rosetta emulation took 2.5+ hours and crashed
- **NEXT STEP**: Build on native x86 Linux machine with correct ROCm 7.1 base image

**Dockerfile** at `/tmp/Dockerfile.runner-exact`:
```dockerfile
FROM ghcr.io/actions/actions-runner:latest
# Installs ROCm 7.1, PyTorch 2.10.0+rocm7.1, AITER, prebuilds all modules
# The runner uses: Torch 2.10.0+rocm7.1, Ubuntu 24.04, Python 3.12
```

**Build script** at `/tmp/build_moe_modules.py` — builds only the 6 needed MOE modules (not all 40+).

**Submission template** at `142_download_precompiled.py`:
```python
import urllib.request
# Downloads .so files from GitHub releases to /home/runner/aiter/aiter/jit/
# BEFORE any AITER imports → skips JIT compilation
```

### Path 2: Triton matmul_ogs MOE Kernel (The Performance Breakthrough)

**Discovery**: AITER contains `moe_gemm_a4w4` — a Triton-based MOE GEMM kernel adapted from `triton_kernels/matmul_ogs` that is fundamentally different from the CK 2-stage path:

**Location**: `aiter/ops/triton/moe/moe_op_gemm_a4w4.py`

**Key advantages**:
- Uses `tl.dot_scaled(x, x_scales, "e2m1", w, w_scales, "e2m1")` — hits native CDNA4 `v_mfma_scale_f32_16x16x128_f8f6f4` instruction directly
- Fuses routing + GEMM + SiLU activation in a SINGLE kernel
- XCD swizzling (`num_xcds=8`) and aggressive tiling (`block_n=512, block_k=256`)
- Eliminates separate sort → quant → GEMM → requant → GEMM pipeline
- No CK module dependency → no 216s JIT compilation

**Status** (v140):
- Code written at `140_pure_triton.py` — architecturally correct
- Weight format solved: `torch.as_strided` creates column-major view without copying
- Fixed config override eliminates Triton kernel variant explosion (was 51 constexpr params)
- Routing construction (`_build_routing_data`) builds `RoutingData` + `ExptData` from `topk_ids`/`topk_weights`
- **BLOCKER**: Triton JIT compilation still takes ~120-200s on cold runners (times out)
- **FIX**: AOT-compile the Triton kernels to `.hsaco` in Docker, download at runtime (same approach as Path 1)

**Key interfaces**:
```python
from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_a4w4
from aiter.ops.triton.moe.moe_routing.routing import RoutingData, ExptData

# Stage 1: x_fp4 × w1 + SiLU fusion
stage1_out = moe_gemm_a4w4(
    x_fp4, w1, x_scales, w1_scales,
    routing_data=routing_data,
    gather_indx=gather_indx,
    scatter_indx=scatter_indx,
    apply_swiglu=True,
    out_dtype=torch.bfloat16,
)
```

**Weight format**: Raw weights are `[E, N, K//2]` fp4x2. Kernel needs `[E, K//2, N]` column-major (`stride(-2)==1`). Solution:
```python
E, N, Kh = w_fp4.shape
wt = torch.as_strided(w_fp4, (E, Kh, N), (N * Kh, 1, Kh))  # zero-copy view
```

**Scale format**: `gate_up_weight_scale` is 2D on the runner (not 3D as task.py suggests). Need reshape before transpose.

**CORRECTNESS NOT YET VERIFIED** — the routing construction and kernel invocation haven't been tested end-to-end on a runner that completes without timeout.

## What Has Been Tried and Failed (Key Learnings)

| Approach | Result | Lesson |
|----------|--------|--------|
| Environment variables (HIP_FORCE_DEV_KERNARG, GPU_MAX_HW_QUEUES, AITER_USE_NT, AMD_DIRECT_DISPATCH) | All neutral | MI355X ignores these |
| block_m tuning for CK shapes | Heuristic is already optimal | `get_block_size_M` minimizes CU idle rounds correctly |
| CK kernel name overrides (256x32 for E=257 bs=512) | Correctness failures | 256x kernel incompatible with N=256 |
| 1-stage ASM kernel (fmoe_g1u1) | Correctness bugs at large batch sizes | Unreliable for benchmark shapes |
| Hybrid CK stage1 + cktile stage2 | RuntimeError "Unsupported kernel config" | ksplit=2 only works with cktile, not CK |
| CSV config injection (cfg_2stages dict patching) | Timeouts on cold runners | Triggers additional CK module loading |
| Direct fused_moe_2stages replacement | block_size alignment assertion | `fused_dynamic_mxfp4_quant_moe_sort` requires block_size % 32 == 0 |
| load_inline C++ compilation | >540s timeout | Runner compilation is too slow |
| gc.disable(), Python micro-optimizations | Neutral | GPU kernel time dominates |
| dispatch_policy=1 for sorting | 33% WORSE for dense shapes | Never use |
| AITER_ONLINE_TUNE=1 | 12-min timeout | Runtime profiling too slow |

## File Layout

```
moe-mxfp4/
├── submission.py              ← Currently v142 (download precompiled + v103)
├── 103_v98_opus_only.py       ← Best reliable baseline (~152µs)
├── 140_pure_triton.py         ← Triton matmul_ogs (correct but times out)
├── 142_download_precompiled.py ← v103 + downloads pre-compiled .so at runtime
├── task.py                    ← Input/output type definitions
├── PROMPT.md                  ← This file
├── research/                  ← Earlier research notes
└── *.py, *.md                 ← 140+ experiment files

/tmp/
├── Dockerfile.runner-exact    ← Docker config matching runner environment
├── build_moe_modules.py       ← Selective build script for 6 MOE modules
└── aiter_so/                  ← Pre-compiled .so files (ROCm 7.2, partial)

GitHub:
└── samuelzxu/aiter-precompiled  ← Hosts .so files for runtime download
    └── releases/v0.1-rocm72/    ← 3 modules (sorting, quant, enum)
```

## Immediate Next Steps (Priority Order)

### 1. Build pre-compiled CK modules on x86 Linux
```bash
# On x86 Linux machine with Docker:
cd /tmp
docker build --platform linux/amd64 -f Dockerfile.runner-exact -t runner-exact .
# Extract .so files:
docker run --rm -v /tmp/aiter_so:/output runner-exact \
  bash -c 'cp /home/runner/aiter/aiter/jit/*.so /output/'
# Upload to GitHub:
gh release create v0.2-rocm71 --repo samuelzxu/aiter-precompiled /tmp/aiter_so/*.so
```

If `setup.py develop` fails on some modules (non-MOE ones), use `build_moe_modules.py` to build only the 6 we need.

### 2. Update v142 submission with all pre-compiled modules
Add all successfully-built `.so` URLs to the `_MODULES` dict in `142_download_precompiled.py`. Test with `--mode test`.

### 3. Test v140 (Triton matmul_ogs) with pre-compiled CK fallback
Once cold runner timeouts are eliminated, the Triton path should have enough time budget for JIT compilation. Or AOT-compile the Triton kernels too.

### 4. Debug v140 correctness
The routing construction (`_build_routing_data`) hasn't been verified. Key concerns:
- Is `gather_indx` correct? (should map sorted position → original token index)
- Is `scatter_indx` correct? (should map sorted position → original flat index)
- Does `block_pid_map` packing match what the kernel expects? (`(bid << 16) | eid`)
- Stage 2 routing: tokens are already gathered from stage 1 — need correct gather/scatter

### 5. AOT-compile Triton kernels
```python
from triton.compiler import compile
from triton.runtime import GPUTarget
target = GPUTarget("hip", "gfx950", 64)
# Compile moe_gemm_a4w4 with specific constexpr values
```

## Key Technical Details

### Runner Environment
- GPU: AMD Instinct MI355X (gfx950, 256 CUs, CDNA4)
- CPU: AMD EPYC 9575F
- ROCm: 7.1
- PyTorch: 2.10.0+rocm7.1
- Ubuntu: 24.04 (noble)
- Python: 3.12
- AITER: installed at `/home/runner/aiter/`
- JIT cache: `/home/runner/aiter/aiter/jit/*.so`
- 12-minute total timeout per submission

### AITER Code Paths
- **CK 2-stage** (current v103): `fused_moe_2stages` → `ck_moe_stage1` + `ck_moe_stage2`
- **cktile** (v103 for sparse): `cktile_moe_stage1` + `cktile_moe_stage2` (BF16 activations, skip quant)
- **Triton matmul_ogs** (v140): `moe_gemm_a4w4` with `tl.dot_scaled` native MXFP4
- **1-stage ASM** (broken): `fmoe_g1u1` (correctness issues, don't use)

### Critical Discovery: cktile skips quant
Line 1083 of `fused_moe.py`: when `q_dtype_a==fp4x2 and ksplit>1 and is_shuffled`, the code passes raw BF16 hidden_states directly to cktile kernels — NO quantization step. This is why cktile is fast for sparse shapes.

### Memory Files
Read `/Users/samuelxu/.claude/projects/-Users-samuelxu-dev-reference-kernels/memory/` for:
- `moe_mxfp4_optimization.md` — Full optimization history
- `competition_rules.md` — Anti-cheating rules
- `feedback_autonomous.md` — User prefers autonomous iteration
- `feedback_no_resubmit.md` — Don't resubmit same version repeatedly
- `precompilation_unblock.md` — Cross-compilation approach
- `submission_command.md` — Submit command reference

## The Path to 114µs

The top competitors are almost certainly using either:
1. **Custom HIP C++ kernels** via `load_inline` with pre-compiled binaries (compressed + embedded)
2. **Triton matmul_ogs** (`moe_gemm_a4w4`) with AOT-compiled `.hsaco` kernels

Our v140 (Triton matmul_ogs) is the right architecture. It uses `tl.dot_scaled` which hits the same native MFMA instruction as the mxfp4-mm preshuffle breakthrough. The blocking issue is purely compilation time — solve that with pre-compilation and we can iterate on kernel tuning to close the gap from 152µs to 114µs.
