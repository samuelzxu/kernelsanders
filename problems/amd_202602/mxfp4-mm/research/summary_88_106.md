# Experiments 88-106: Comprehensive Optimization Summary

## Final Best: #102 at ~16.0µs ranked geomean

### Improvements over original baseline (~24µs reference):
1. Config injection for all 6 benchmark shapes (biggest win)
2. Fused quant+GEMM for M<=16 KSPLIT=1 (saves kernel launch)
3. BSN=64 for K=2048 (-1.2µs confirmed in #90/#91/#92)
4. MFMA 32x32 for M=256 K=1536 (-0.5µs confirmed in #101/#102)
5. data_ptr caching for B_scale
6. e8m0_unshuffle for K>512 (avoids double-quant)
7. torch.inference_mode()

### Per-shape breakdown (ranked):
| Shape | Time | Bottleneck |
|-------|------|-----------|
| M=4, N=2880, K=512 | ~12.4µs | B_scale quant overhead |
| M=16, N=2112, K=7168 | ~21.2µs | GEMM compute (KSPLIT=8) |
| M=32, N=4096, K=512 | ~12.7µs | Low CU occupancy (64 tiles) |
| M=32, N=2880, K=512 | ~12.6µs | Low CU occupancy (45 tiles) |
| M=64, N=7168, K=2048 | ~20.2µs | GEMM compute |
| M=256, N=3072, K=1536 | ~20.0µs | GEMM compute + reduction |

### All experiments tried:
| # | Approach | Result | Why |
|---|----------|--------|-----|
| 88 | CK ASM kernels | 2x slower | Terrible default configs |
| 89 | gemm_a16wfp4 fused | Neutral | No remap_xcd |
| 90 | Config sweep | BSN=64 works | K=2048 +1.3µs |
| 91 | Fused M<=32 | Worse | Register pressure |
| 92 | **Combined best** | **16.0µs** | BSN=64 + best configs |
| 93 | gemm_a16wfp4 for KSPLIT>1 | Neutral | No remap_xcd |
| 94 | BSK=1024 K=1536 | NaN | EVEN_K masking bug |
| 95 | Fused splitK + xcd | Trades M=4↔M=16 | JIT overhead |
| 96 | Hybrid fused | Same trade | Extra kernel defs |
| 97 | e8m0_unshuffle all K | Worse | Slower for K=512 |
| 98 | Config/output pre-pass | Much worse | Breaks gemm internals |
| 99 | Preshuffle variant | JIT bug | b undefined |
| 100 | Gluon variant | AssertionError | Incompatible Triton |
| 101 | MFMA32 all shapes | Mixed | Hurts M=64 |
| 102 | **MFMA32 M=256 only** | **-0.5µs** | Helps M=256 only |
| 103 | MFMA32 M=64+M=256 | Worse | Hurts M=64 confirmed |
| 104 | num_warps=8 K=2048 | Neutral | No benefit |
| 105 | Hardcoded dispatch | Neutral | Python overhead negligible |
| 106 | Patched preshuffle | Neutral | Scale shuffle overhead |

### Key learnings:
- remap_xcd is essential (40% impact on MI355X)
- MFMA 32x32 only helps M>=128 with BSM>=64
- Preshuffle has JIT bug AND scale overhead that negates gains
- CK ASM defaults are terrible (need per-shape tuning)
- Config tuning has diminishing returns beyond shape-specific injection
- Python overhead (dict lookup, lambda) is negligible vs GPU time
- The ~3µs gap to leader requires fundamentally different kernel code
