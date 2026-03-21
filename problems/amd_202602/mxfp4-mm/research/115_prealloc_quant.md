# #115 Pre-allocated Quant Buffers

## Approach
Call `_dynamic_mxfp4_quant_kernel` directly with pre-allocated A_q and A_scale
buffers to avoid `torch.empty` allocation on every call.

## Results
No measurable improvement. GPU tensor allocation via `torch.empty` is ~0.1µs
on AMD GPUs (much faster than the 0.5µs estimate). Saving two allocations per
call is ~0.2µs, within measurement noise.

The actual bottleneck breakdown for M>16 shapes:
- GEMM compute: ~18-19µs (dominates)
- Quant compute: ~1.5µs
- Quant allocation: ~0.2µs (negligible)
- Python overhead: ~0.5µs
