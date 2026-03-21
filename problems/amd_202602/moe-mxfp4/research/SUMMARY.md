# MoE MXFP4 Optimization Summary

## Final Result
**Baseline (AITER fused_moe) is optimal, achieving ~1.04-1.31x speedup vs reference.**

Run-to-run variance: ~3-5% (normal for GPU benchmarks)

## Attempts Summary
| # | Description | Result |
|---|-------------|--------|
| 1 | Baseline (AITER fused_moe) | ✅ ~1.14x geomean speedup |
| 2 | doweight_stage1=True | ❌ Failed correctness tests |
| 3 | Shared expert separation | ⏸️ Complex, AITER handles internally |
| 4 | Environment variables | ❌ 1-5% slower |
| 5 | asm_moe investigation | ❌ Not MXFP4 compatible |
| 6 | contiguous() call | ✅ No significant difference |
| 7 | Shared expert as dense GEMM | ⏸️ Complex weight extraction |
| 8 | torch.compile | ❌ Pickle error with multiprocessing |
| 9 | Remove contiguous() | ✅ Same performance |

## Latest Results (Ranked Benchmark)
| bs | E | d_expert | Time [µs] | Reference [µs] | Speedup |
|----|---|----------|-----------|----------------|---------|
| 16 | 257 | 256 | 137 | 152.7 | 1.11x |
| 128 | 257 | 256 | 224 | 239.0 | 1.07x |
| 512 | 257 | 256 | 257 | 336.5 | 1.31x |
| 16 | 33 | 512 | 95.5 | 106.2 | 1.11x |
| 128 | 33 | 512 | 130 | 141.1 | 1.09x |
| 512 | 33 | 512 | 216 | 225.0 | 1.04x |
| 512 | 33 | 2048 | 350 | 380.4 | 1.09x |

**Geometric Mean Speedup: ~1.11x**

## Key Insights
1. AITER's `fused_moe` is highly optimized for MXFP4 MoE
2. Auto-tuned kernel selection based on problem dimensions
3. Pre-tuned kernels for DeepSeek-V3/R1 configurations (E=257)
4. No user-tunable parameters exposed

## Why We Can't Beat the Baseline
1. **doweight_stage1** - Correctness requirement, not tuning
2. **asm_moe** - FP8-only, not MXFP4 compatible
3. **Environment variables** - Interfere with auto-tuning
4. **torch.compile** - Incompatible with multiprocessing eval
5. **Kernel internals** - block_m, splitk auto-selected by AITER
6. **Lower-level APIs** - Not exposed (ck_moe_stage1/2 are internal)

## Current Submission
Simple AITER fused_moe call:
```python
output = fused_moe(
    hidden_states,
    gate_up_weight_shuffled, down_weight_shuffled,
    topk_weights, topk_ids,
    activation=ActivationType.Silu,
    quant_type=QuantType.per_1x32,
    doweight_stage1=False,
    w1_scale=gate_up_weight_scale_shuffled,
    w2_scale=down_weight_scale_shuffled,
    hidden_pad=hidden_pad,
    intermediate_pad=intermediate_pad,
)
```

## Further Improvements Would Require
- Custom Triton/CK kernels from scratch (significant effort)
- Changes to AITER library itself
- Different quantization format (not problem spec compatible)
