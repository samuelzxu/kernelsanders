# MOE-MXFP4 Competition Context

## Problem
Mixture-of-Experts GEMM layer with MXFP4 quantization, modeling DeepSeek-R1 MoE.
- Input: hidden_states [M, 7168], expert weights in FP4 (gate_up + down), top-k routing
- Architecture: top-8 routed experts + 1 shared expert = 9 total, up to 257 experts
- Each expert: gate_up GEMM → SiLU activation → down GEMM
- Output: [M, 7168] bfloat16

## Benchmark Shapes (7 configurations)
| # | bs | E (experts) | d (inter_dim) | tok/expert | Sparsity |
|---|-----|-------------|----------------|------------|----------|
| 1 | 16  | 257         | 256            | 0.50       | Very sparse |
| 2 | 128 | 257         | 256            | 3.99       | Sparse |
| 3 | 512 | 257         | 256            | 15.9       | Moderate |
| 4 | 16  | 33          | 512            | 3.88       | Sparse |
| 5 | 128 | 33          | 512            | 31.0       | Moderate |
| 6 | 512 | 33          | 512            | 124.1      | Dense |
| 7 | 512 | 33          | 2048           | 124.1      | Dense, large K |

## Competition Submission Commands

```bash
popcorn submit --mode test --no-tui submission.py      # correctness (10/h)
popcorn submit --mode benchmark --no-tui submission.py  # performance (10/h)
popcorn submit --mode leaderboard --no-tui submission.py # ranked (1/h)
```

**Headers required:**
```python
#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
```

**BANNED WORD:** "stream" — server rejects submissions containing it.

**Anti-cheating:** No cross-invocation caching. Within-call preprocessing OK. Benchmark clears L2 cache between iterations (cold cache measurement).
Reference: https://gist.githubusercontent.com/hargup/4897f33df0d2425ac4c8c99dc8f6ec00
Reference: https://deep-reinforce.com/defense_kernel_hack.html

## Key Libraries on Runner

- **aiter**: `from aiter.fused_moe import fused_moe` — main MoE dispatch kernel
  - `fused_moe()` with FP4 quantized weights, SiLU activation
  - CK 2-stage modules at `/home/runner/aiter/hsa/gfx950/fmoe_2stages/`
  - ASM 1-stage modules at `/home/runner/aiter/hsa/gfx950/fmoe/`
  - `aiter.ops.triton.gemm` — Triton GEMM kernels
  - `aiter.ops.shuffle.shuffle_weight()` — preshuffle for CK ASM path
- **Triton 3.6.0**: Custom Triton kernels, `tl.dot_scaled` for FP4
- **PyTorch 2.10.0+rocm7.1**: `load_inline` for JIT HIP compilation
- **ROCm 7.1**: hipcc, rocBLAS, hipBLASLt at `/opt/rocm/`
- **FlyDSL v0.0.1.dev**: Available on runner. FlyDSL stage 2 kernels for afp4_wfp4.

## Docker Cross-Compilation

```bash
docker run --rm -v $(pwd):/workspace rocm/dev-ubuntu-24.04:7.1-complete bash -c "
  pip install --break-system-packages pybind11 torch==2.10.0+rocm7.1 --index-url https://download.pytorch.org/whl/rocm7.1
  cd /workspace && hipcc -O3 -std=c++20 --offload-arch=gfx950 -mcumode ...
"
```

**NOTE:** Only ONE Docker build at a time (RAM limit). Cross-compiled .co may fail with `hipErrorInvalidImage` — use `load_inline` (JIT on runner) as reliable alternative.

## Key Runner Paths

```
/home/runner/aiter/                           # aiter installation
/home/runner/aiter/hsa/gfx950/fmoe/          # ASM 1-stage MoE kernels
/home/runner/aiter/hsa/gfx950/fmoe_2stages/  # CK 2-stage MoE kernels
/home/runner/aiter/hsa/gfx950/f4gemm/        # FP4 GEMM CK ASM kernels
/opt/rocm/include/ck, ck_tile, rocwmma       # CK headers
```

## MI355X Architecture
- gfx950 / CDNA4, 304 CUs, 8 XCDs
- 6.4 TB/s HBM, 10 PFLOP/s FP4, 2.5 PFLOP/s BF16
- FP4 MFMA: `__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4`
- LDS: 64KB per CU, 64 banks with ds_read_b128
- AMD swizzles on HBM addresses, NOT LDS

## Current Best Approach (v185, ~146us geomean)
Hybrid dispatch with 4 kernel backends selected per-shape:
1. **cktile** (BF16 path, no quant) for sparse shapes (tok/expert < 40)
2. **ASM 1-stage** for E<=33, d<=512 (fused gate+up+down, no intermediate)
3. **CK stage1 + FlyDSL atomic stage2** for d>1024 (E=33 d=2048)
4. **CK 2-stage** with DSv3 CSV tuning for E=257 dense shapes

Plus: 8 pre-compiled AITER JIT modules (eliminates 308s cold start), env var tuning (NT=1, HW_QUEUES=2), FlyDSL cfg_2stages injection via monkeypatch.

---

## Optimization History & Key Findings

### 1. AITER fused_moe Dispatch Strategies

**CK 2-stage (default path):** Sort → FP4 quant → GEMM1 → requant → GEMM2. The baseline approach. Uses pre-tuned CSV configs from DSv3. Geomean ~156us with default settings, ~178us without cktile overrides (bare default is 17% worse).

**CK cktile (BF16 path):** Operates in BF16 (skips FP4 quantization entirely). The single most impactful optimization: -27us geomean vs default. Best for sparse shapes where few tokens per expert means quant overhead dominates. Uses split_k=2 for very sparse (tok/exp<5), split_k=1 for moderate (tok/exp<40). Block_m=16 for small token counts, 32 for medium.

**ASM 1-stage:** Fuses both GEMM stages + SiLU into one kernel launch. Pre-compiled .co files at `hsa/gfx950/fmoe/silu/`. Only has M=32 tiles (32x256 2tg, 32x512 1tg). 30% slower globally than CK 2-stage due to suboptimal tile configs for most shapes. Wins marginally for E<=33 d<=512 shapes. Correctness: near-pass (2-3 elements off on some shapes).

**FlyDSL stage 2:** Assembly-tuned MFMA kernels injected via cfg_2stages monkeypatch. Only stage 2 is available (no stage 1). Valid configs for afp4_wfp4:
- `t32x256x256_reduce` — 371us (worst)
- `t64x256x256_reduce` — 330us
- `t64x256x256_atomic` — 320us (BEST, 7.2% better than CK for shape 7)
- `t64x128x128_atomic` — invalid
- `t128x256x256_reduce` — invalid

Atomic mode is faster than reduce mode. FlyDSL helps for d=2048 but is 1.6% worse for d=256 shapes.

**1-stage ASM variants tested:**
- `fmoe_bf16_pertokenMXfp4_g1u1_vs_silu_1tg_ps_32x512` (default auto-select)
- `novs` (no vector store) variants: not registered in dispatch table, can't force
- Auto-select is already optimal among registered variants

### 2. Docker Pre-compilation of JIT Modules

Built all 8 AITER modules via Docker (ROCm 7.1, gfx950 target):
1. `module_aiter_enum.so` (524KB) — saves ~12s
2. `module_moe_sorting_opus.so` (1MB) — saves ~27s
3. `module_moe_sorting.so` (1.2MB)
4. `module_quant.so` (2.6MB) — saves ~25s
5. `module_activation.so` (1MB) — saves ~22s
6. `module_moe_cktile2stages.so` (17MB)
7. `module_moe_ck2stages_fp4x2_fp4x2_preshuffle_on_b16_silu_per_1x32_mulWeightStage2_.so` (1.9MB)
8. `module_moe_asm.so` — saves ~33s

Total: ~308s cold start eliminated. Hosted on GitHub (samuelzxu/aiter-precompiled v0.3-rocm71). Downloaded via urllib at submission startup. No performance improvement (only startup time).

### 3. Triton MoE Kernels

**tl.dot_scaled GEMM (v165):** Custom minimal Triton kernel confirmed tl.dot_scaled works perfectly with AITER's fp4x2 weights — max_err=0.000000. The critical discovery: weight scales must be [N, K//32] (N-major), NOT [K//32, N]. All previous Triton failures (v152-v154) were from transposed scales.

**fused_moe_mxfp4_silu (v143-v154, v166-v177):** AITER's Triton MoE kernel. Initially produced -1e29 overflow values. Root cause: interleaving code in early attempts corrupted scale data; the raw scale format [E, N, K//32] is correct. With correct scales (v166), kernel executes with reasonable values. However:
- BF16 activations: kernel works but output mismatches FP4-quantized reference (skipping quant = different precision)
- FP4 activations: GPU memory fault. The kernel accesses A_mx_scale at sorted_token_ids positions; AITER's sorting padding entries have huge values (~150M), causing OOB reads. Fixed with clamp (v176) but full pipeline still fails correctness.
- Hybrid Triton stage1 + CK stage2: layout mismatch (sorted order vs token order)

**matmul_ogs / moe_gemm_a4w4 (v152):** Triton grouped GEMM approach. Stage 1+2 execute but correctness fails. Debugged extensively: nibble swap not the issue, scale strides verified, weight layout verified. 73% of elements wrong = systematic error in activation quantization format or routing. Required local GPU debugging.

**triton_kernels package:** Not installed on runner, pip install fails.

### 4. Shape-Specific Dispatch Policies

The dispatch policy in the current submission:
```
tok/expert < 5  → cktile, split_k=2 (very sparse, E=257 bs=16)
tok/expert < 40 → cktile, split_k=1 (sparse/moderate)
E<=33, d<=512   → ASM 1-stage (fused pipeline)
d > 1024        → CK stage1 + FlyDSL atomic stage2
E=257 dense     → CK 2-stage with DSv3 CSV tuning
```

Key thresholds tested:
- cktile for tok/exp < 64 (AITER default): too aggressive, moderate shapes suffer
- cktile for ALL shapes: 182us geomean (worse)
- cktile for E=257 bs=512 (tok/exp=15.9): 16% SLOWER than CK (v100)
- ASM for d>512: much slower (d=2048: 505us vs 330us CK+FlyDSL)
- ASM for E=257: slower than CK 2-stage for all E=257 shapes

### 5. Split-K Tuning

| Shape | sk=1 | sk=2 | sk=3 | sk=4 | Best |
|-------|------|------|------|------|------|
| Sparse (tok/exp<5) | - | Better | - | - | sk=2 |
| Moderate (tok/exp<40) | Better | - | - | - | sk=1 |
| Dense E=33 | CK default | 3x worse for d=2048 | - | - | CK default |
| Global KSPLIT=2 | - | +20% worse | - | - | Per-shape |

AITER_KSPLIT=2 globally: catastrophic for dense shapes (d=2048 goes 352→1024us).
ksplit=3 for cktile: tested, no improvement over sk=1/sk=2 split.

### 6. Environment Variable Tuning

| Variable | Value | Effect |
|----------|-------|--------|
| `AITER_USE_NT=1` | Force non-transpose | +0.4% vs heuristic (marginally better) |
| `HIP_FORCE_DEV_KERNARG=1` | Device kernel args | Required for kernel arg handling |
| `GPU_MAX_HW_QUEUES=2` | HW queue count | Same as Q=4, keeping Q=2 |
| `AITER_BYPASS_TUNE_CONFIG` | Skip CSV | Worse (shape 3: 246→304us) |
| `AITER_KSPLIT=2` | Global split-K | +20% worse (d=2048 3x slower) |

### 7. HIP Direct Launch Approaches

**HIP module launch (v49-v50):** Attempted to launch pre-compiled .co ASM kernels directly via hip-python, bypassing the module_moe_asm C++ wrapper. Would save ~33s JIT. Blocked by complex KernelArgs struct layout.

**HIP graphs (v190):** Attempted to capture sort→quant→stage1→requant→stage2 pipeline into a graph. Blocked: multi-stream not allowed in submission environment.

**torch.compile (v48, v191):** Can't trace through opaque CK/ASM kernel calls. Also pickle error in multiprocessing eval harness.

### 8. Sorting/Routing Optimizations

**moe_sorting_opus_fwd:** Faster sorting kernel (opus variant). Used when available. Pre-compiled module uploaded.

**Pre-allocated sorting buffers:** Cached per (M, E, model_dim, block_size_M). Useful with manual fused_moe_2stages call but unnecessary with high-level fused_moe (which uses fused_dynamic_mxfp4_quant_moe_sort internally).

**fused_dynamic_mxfp4_quant_moe_sort:** Fused kernel combining quantization + sorting in one launch. Used internally by fused_moe high-level dispatch. Saves ~2-4us vs separate quant + sort.

**Sorting dispatch policy:** moe_sorting_dispatch_policy=1 tested, no significant impact.

### 9. Custom Kernel Attempts

**Custom Triton MOE GEMM (v46):** Full custom kernel with tl.dot_scaled, expert-parallel dispatch. Correctness issues with routing data construction.

**Skip-requant approaches (v70, v155, v160):** Attempted to eliminate intermediate FP4 requantization between stages. All blocked: cktile BF16 stage2 too slow for dense, CK a16w4 stage2 "Unsupported kernel config" for many shapes, manual stage splitting has arg mismatches.

**Raw weight (BNS) path (v197):** preshuffle_off module auto-compiled on runner (68.6s). Wrong results — raw scales incompatible with BNS kernel expectations.

**doweight_stage1=True (v156):** Apply routing weights in stage 1. GPU memory access fault (null pointer crash). Fundamentally broken for these shapes.

### 10. Block_m Tuning

Shape 7 (bs512, E33, d2048) sweep:
- block_m=32: 408us (worst — too many small blocks)
- block_m=64: 327us (BEST)
- block_m=128: 343us (AITER default heuristic)

Cktile block_m: 16 for small token counts (<2048 padded), 32 for medium.

### 11. Direct Pipeline (Bypassing fused_moe)

v189: Called sort→quant→sort_scales→CK_stage1→requant→sort_scales→CK_stage2 directly with pre-allocated buffers. Saved ~4us per CK 2-stage call. Python overhead in fused_moe is only ~4-5us, not the 15-20us estimated. Not worth the complexity.

---

## Per-Shape Best Timings

Best measured times across all experiments (v185 leaderboard with FlyDSL atomic):

| # | Shape | Best Time | Kernel Backend | Notes |
|---|-------|-----------|----------------|-------|
| 1 | bs=16, E=257, d=256 | 86-87us | cktile (sk=2) | Very sparse, BF16 path |
| 2 | bs=128, E=257, d=256 | 169-171us | cktile (sk=1) | Sparse, BF16 path |
| 3 | bs=512, E=257, d=256 | 239-248us | CK 2-stage (CSV) | Moderate density |
| 4 | bs=16, E=33, d=512 | 56-58us | cktile (sk=2) | Sparse, BF16 path |
| 5 | bs=128, E=33, d=512 | 105-107us | cktile/ASM | Moderate density |
| 6 | bs=512, E=33, d=512 | 205-211us | ASM 1-stage | Dense |
| 7 | bs=512, E=33, d=2048 | 320-330us | CK+FlyDSL atomic | Dense, large K |
| **Geomean** | | **~146us** | | **Rank ~#11/70** |

Target: #1 at 114.6us (22% gap).

Progression: baseline ~178us → cktile ~156us → v148 zero-JIT ~151us → v161 fused dispatch ~151us → v175 block_m tuning ~148us → v185 FlyDSL atomic ~146us.

---

## Dead Ends

| Approach | Why it failed |
|----------|---------------|
| **Swiglu activation (v151)** | Different math than SiLU reference — fundamentally incompatible |
| **Triton fused_moe_mxfp4 with BF16 acts** | Produces more-accurate results that FAIL 2% tolerance (precision mismatch vs FP4 reference) |
| **Triton fused_moe_mxfp4 with FP4 acts** | GPU memory fault — sorted_token_ids padding has huge values (~150M) causing OOB scale reads |
| **Triton matmul_ogs (moe_gemm_a4w4)** | 73% elements wrong — systematic error in activation quantization format |
| **AITER_KSPLIT=2 global** | d=2048 shape goes 352→1024us (+3x), BF16 intermediate too large |
| **doweight_stage1=True** | GPU null pointer crash on MI355X |
| **cktile for ALL shapes** | 182us geomean — worse for moderate/dense shapes |
| **ASM 1-stage for all** | 203us geomean — M=32 tiles not tuned for most shapes |
| **ASM 1-stage for d=2048** | 505us vs 330us CK+FlyDSL — 53% slower |
| **HIP graphs** | Multi-stream not allowed in submission environment |
| **torch.compile** | Can't trace opaque CK/ASM kernels; pickle error in eval harness |
| **triton_kernels package** | Not installed on runner, pip install fails |
| **FlyDSL t128/t32 tiles** | Invalid for afp4_wfp4; only t64x256x256 works |
| **FlyDSL for d=256 shapes** | 1.6% worse than CK for E=257 |
| **Raw weight BNS path** | Scale format incompatible with BNS kernels |
| **ASM novs variants** | Not registered in dispatch table, can't force |
| **Bypass tuned CSV** | 24% worse for E=257 shapes (304us vs 246us) |
| **Skip-requant (BF16 intermediate)** | CK a16w4 stage2 "Unsupported config" for most shapes; cktile BF16 stage2 3x slower for dense |
| **Pre-allocated buffers** | PyTorch caching allocator already handles this; negligible gain |
| **GPU_MAX_HW_QUEUES=4** | No difference from Q=2 |

---

## Remaining Opportunities

### Partially Explored (need local GPU)
1. **Full Triton MOE kernel with correct FP4 pipeline:** tl.dot_scaled proven correct (v165). The remaining blocker is activation scale indexing (sorted_token_ids OOB). With local GPU debugging, could clamp/pad scales properly and build a complete Triton pipeline that fuses quant+GEMM+SiLU.

2. **Triton stage1 + requant + CK stage2:** Match CK's precision by requantizing Triton's BF16 stage1 output to FP4 before passing to CK stage2. Would need correct sort-order handling.

3. **Custom per-shape tuned CSV:** AITER's gemm_moe_tune.py can generate per-shape optimized kernel configs. Requires running the tuner on MI355X hardware.

### Unexplored
4. **Custom HIP/ASM kernels:** Hand-optimized MFMA instruction scheduling, fused sort+quant+GEMM+SiLU+GEMM pipeline. The 22% gap to #1 likely requires this level of optimization.

5. **Pre-compiled .hsaco binaries:** Embed custom compiled kernel binaries directly in submission, bypassing all JIT and AITER overhead.

6. **FP8 intermediate path:** Use FP8 (e5m2/e4m3) for intermediate instead of FP4 requant. Less precision loss than BF16, potentially faster than FP4 requant.

7. **Expert-level pipelining:** Overlap GEMM computation across different experts within a single kernel launch.

8. **Weight reformatting at load time:** Pre-process weights into a format that eliminates scale sorting overhead.

9. **FlyDSL stage 1:** Currently only stage 2 is available. If AITER adds stage 1 FlyDSL support for afp4, it could improve shapes 1-5.

---

## Key Technical Insights

### HipKittens Paper Architecture Patterns ([paper](https://arxiv.org/abs/2511.08083), [code](https://github.com/HazyResearch/HipKittens))
These patterns are critical for anyone writing custom HIP/ASM kernels on MI355X:

- **8-Wave Ping-Pong:** 2 waves per SIMD alternate compute (MFMA only) and memory (loads only) roles, controlled by conditional `s_barrier`. This is how CK ASM achieves peak performance. Enables 256x256 output tiles.
- **4-Wave Interleave:** 1 wave per SIMD with fine-grained instruction staggering. Better for imbalanced workloads (e.g., attention backward).
- **Pin Register:** Explicitly map tiles to physical VGPR/AGPR registers, bypassing HIPCC. Enables AGPRs as direct MFMA inputs (compiler adds unnecessary `v_accvgpr_read` moves otherwise).
- **Chiplet-Aware Scheduling:** XCD grouping + windowed traversal. Vertical windows of height 4-8. Target ~79% L2 hit rate, 55-93% LLC. L2 BW ~3x LLC BW. 15-19% improvement on GEMMs.
- **LDS Banks:** `ds_read_b128` = 64 banks; `ds_read_b96` = 32 banks in 8 non-sequential phases. Phase ordering is undocumented — requires empirical solver.
- **HBM Swizzle, NOT LDS:** Conflict-free LDS access is achieved by swizzling global memory addresses during HBM→LDS transfer, not by swizzling LDS addresses.
- **Static Registers:** 512 regs per SIMD (256 VGPR + 256 AGPR) statically divided across waves. Wave specialization loses tile size — ping-pong (same code both waves) is better.
- **Performance:** HipKittens matches AITER assembly on BF16/FP8 GEMM. 1.8-2.4x over baselines on GQA backward. 1.3-3.0x over Triton.

### MoE on MI355X
- **Sparsity is the key variable.** For sparse shapes (few tokens per expert), the FP4 quantization overhead dominates the actual GEMM. cktile BF16 path (no quant) wins by 30-40% per shape.
- **Dense shapes are GEMM-bound.** For shape 7 (d=2048, 124 tok/expert), the GEMM itself is the bottleneck. FlyDSL atomic assembly beats CK by 7%.
- **L2 cache is cleared between iterations.** All benchmark measurements are cold-cache. Memory-efficient FP4 format matters for bandwidth-bound shapes.

### AITER Internals
- `fused_moe` (high-level) uses `fused_dynamic_mxfp4_quant_moe_sort` — a fused kernel combining quant + sort. ~2-4us faster than separate calls.
- CK 2-stage pipeline: sort → quant_A → sort_scales → CK_stage1(FP4xFP4) → requant → sort_scales → CK_stage2(FP4xFP4)
- cktile path: sort → cktile_stage1(BF16xFP4, split_k) → cktile_stage2(BF16xFP4). No activation quantization needed.
- ASM 1-stage: sort → fmoe_1stage(BF16_in, FP4_weights, SiLU fused). Single kernel does everything but only has M=32 tiles.
- `get_2stage_cfgs` is the dispatch function. Monkeypatching it via `@functools.lru_cache` wrapper allows per-shape kernel selection.
- `cfg_2stages` dict holds tuned kernel configs. Can inject FlyDSL kernel names here.

### Weight Format
- Weights: `torch.float4_e2m1fn_x2` (two FP4 values packed per byte), shape [E, N, K//2]
- Scales: `torch.float8_e8m0fnu` (e8m0 block scales), shape [E, N, K//32]
- Shuffled weights have different byte layout optimized for CK/ASM MFMA access patterns
- For Triton `tl.dot_scaled("e2m1")`: scales must be N-major [N, K//32], NOT [K//32, N]

### Sorting
- AITER sorting pads expert blocks to block_size_M alignment
- Padding entries in sorted_token_ids have HUGE values (~150M) — causes OOB if used to index into activation tensors
- This is the root cause of Triton fused_moe_mxfp4 GPU memory faults with FP4 activations

### Performance Breakdown
- Python dispatch overhead: ~4-5us per fused_moe call
- FP4 quantization + scale sorting: ~10-20us per stage (the cost cktile avoids)
- GEMM compute: shape-dependent, 40-300us
- The CK 2-stage approach does 4 kernel launches (quant, stage1, requant, stage2) vs 1 for ASM 1-stage, but CK's better tile configs more than compensate

### Competition Context
- 200+ experiments across ~30 hours of optimization
- Rank: ~#11 out of 70 participants
- Best geomean: ~146us, target #1: 114.6us (22% gap)
- Deadline: April 6, 2026
- The gap to #1 almost certainly requires custom kernel development (HIP/ASM with hand-optimized MFMA scheduling) rather than further AITER dispatch tuning
