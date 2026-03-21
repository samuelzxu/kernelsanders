# Attempt 1: Baseline AITER fused_moe

## What we tried
Direct usage of AITER's `fused_moe` kernel with default parameters (same as reference implementation).

## Code
```python
output = fused_moe(
    hidden_states,
    gate_up_weight_shuffled,
    down_weight_shuffled,
    topk_weights,
    topk_ids,
    expert_mask=None,
    activation=ActivationType.Silu,
    quant_type=QuantType.per_1x32,
    doweight_stage1=False,
    w1_scale=gate_up_weight_scale_shuffled,
    w2_scale=down_weight_scale_shuffled,
    a1_scale=None,
    a2_scale=None,
    hidden_pad=hidden_pad,
    intermediate_pad=intermediate_pad,
)
```

## Parameters used
- `activation`: ActivationType.Silu (correct for DeepSeek-R1)
- `quant_type`: QuantType.per_1x32 (MXFP4 block scaling with 32-element blocks)
- `doweight_stage1`: False (apply router weights post-computation)
- Pre-shuffled weights with layout=(16,16)
- Pre-shuffled e8m0 scales

## Expected reference performance
| bs | E | d_hidden | d_expert | top_k | time[us] |
|----|---|----------|----------|-------|----------|
| 16 | 257 | 7168 | 256 | 9 | 152.7 |
| 128 | 257 | 7168 | 256 | 9 | 239.0 |
| 512 | 257 | 7168 | 256 | 9 | 336.5 |
| 16 | 33 | 7168 | 512 | 9 | 106.2 |
| 128 | 33 | 7168 | 512 | 9 | 141.1 |
| 512 | 33 | 7168 | 512 | 9 | 225.0 |
| 512 | 33 | 7168 | 2048 | 9 | 380.4 |

## Result (Ranked Benchmark)
| bs | E | d_expert | Time [µs] | Reference [µs] | Speedup |
|----|---|----------|-----------|----------------|---------|
| 16 | 257 | 256 | 131 | 152.7 | 1.17x |
| 128 | 257 | 256 | 218 | 239.0 | 1.10x |
| 512 | 257 | 256 | 251 | 336.5 | 1.34x |
| 16 | 33 | 512 | 90.5 | 106.2 | 1.17x |
| 128 | 33 | 512 | 128 | 141.1 | 1.10x |
| 512 | 33 | 512 | 215 | 225.0 | 1.05x |
| 512 | 33 | 2048 | 346 | 380.4 | 1.10x |

**Geometric Mean Speedup: ~1.14x vs reference**

## Analysis
Baseline is already faster than reference! Key observations from AITER logs:
- Uses 2-stage pipeline with pre-tuned kernels for E=257 cases
- Dynamic block_m selection for E=33 cases (32, 64, or 128)
- Uses non-temporal load (use_nt) optimization for larger batches
- Kernel names indicate: `MulABScaleShuffled_v3` for stage1, `MulABScaleExpertWeightShuffled_v1` for stage2
