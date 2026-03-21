# MXFP4-MM Optimization: Complete Summary (Experiments 88-116)

## Best Submission: #102 at ~16.0µs ranked geomean
Improvement: ~24µs baseline → ~16µs (33% faster)

## Architecture
```
M<=16, KSPLIT=1: custom fused bf16→fp4 quant + GEMM kernel (single launch)
M>16 or KSPLIT>1: dynamic_mxfp4_quant(A) + gemm_afp4wfp4(A_q, B_q, A_s, B_s)
B_scale: cached by data_ptr. K<=512: re-quant(B). K>512: e8m0_unshuffle(B_scale_sh)
```

## Config Injection Summary
| Shape Key | M Tier | BSM | BSN | BSK | KSPLIT | MFMA | Actual KSPLIT |
|-----------|--------|-----|-----|-----|--------|------|---------------|
| N=2880-K=512 | M<=4 | 16 | 64 | 512 | 1 | 16 | 1 |
| N=2880-K=512 | M<=32 | 32 | 64 | 512 | 1 | 16 | 1 |
| N=4096-K=512 | M<=32 | 32 | 64 | 512 | 1 | 16 | 1 |
| N=2112-K=7168 | M<=16 | 16 | 64 | 256 | 8 | 16 | 7* |
| N=7168-K=2048 | M<=64 | 32 | 64 | 512 | 1 | 16 | 1 |
| N=3072-K=1536 | M<=64 | 32 | 64 | 512 | 2 | 16 | 1* |
| N=3072-K=1536 | M<=256 | 64 | 64 | 512 | 2 | 32 | 1* |
*get_splitk() adjusts KSPLIT (this is optimal, don't fight it)

## Key Findings
1. **BSN=64 for K=2048**: -1.2µs (confirmed across 4 submissions)
2. **MFMA 32x32 for M=256**: -0.5µs (only helps BSM>=64)
3. **Fused quant saves ~2µs** but only for M<=16 (M>=32: 4x more A data = worse)
4. **get_splitk() is correct**: it reduces KSPLIT when reduction overhead exceeds occupancy gain
5. **More tiles ≠ faster**: per-tile work must be substantial to amortize setup
6. **Triton JIT overhead**: each new @triton.jit costs ~3-6µs first-call compilation
7. **GPU tensor alloc**: only ~0.1µs, negligible vs kernel compute

## Dead Ends (verified experimentally)
| Category | Approach | Issue |
|----------|----------|-------|
| Alt kernels | CK ASM (#88) | 2x slower, bad default configs |
| Alt kernels | gemm_a16wfp4 (#89,93) | No remap_xcd |
| Alt kernels | Preshuffle (#99) | JIT bug: b undefined |
| Alt kernels | Gluon (#100) | AMDMFMALayout assert |
| Fused | Split-K + xcd (#95-96) | JIT overhead trades M=4↔M=16 |
| Fused | M>=32 + xcd (#113) | 4x more A data = catastrophic |
| Fused | Atomic add (#112) | bf16 precision loss |
| Overhead | HIP graphs (#109) | GPU sync barriers |
| Overhead | torch.compile (#110) | Pickle error |
| Overhead | Direct kernel (#107) | Wrong SPLITK params |
| Overhead | Pre-alloc (#98,115) | Negligible alloc overhead |
| Config | BSK<K (#108) | More iters = slower |
| Config | KSPLIT=3 (#114) | Reduction dominates |
| Config | num_warps=8 (#104) | No improvement |
| Config | MFMA32 M<=64 (#101,103) | Hurts M=64 |
| Env | HIP_FORCE_DEV_KERNARG (#111) | Already set or no effect |
| Scale | Custom scale kernel (#116) | JIT overhead +6µs |

## Bottleneck Analysis
The ~3µs gap to leader (~13µs) is in the GEMM compute itself:
- M=16 K=7168: 21µs (12x over theoretical 1.6µs)
- M=64 K=2048: 20µs (12x over theoretical 1.6µs)
- Closing this gap requires custom HIP/MFMA kernels with hand-optimized
  LDS tiling, double buffering, and 8-wave ping-pong scheduling.
