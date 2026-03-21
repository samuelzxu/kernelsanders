# Attempt 56: a8w8 everywhere + inference_mode

## Changes
- Always use fp8 Q + fp8 KV (a8w8) matching reference exactly
- torch.inference_mode() decorator for faster tensor ops
- No persistent state across calls
- Fresh metadata each call

## Rationale
Previous attempts with bf16 paths caused correctness failures in ranked
mode because the bf16 kernel (a16w16) produces slightly different numerical
results than the reference's a8w8 kernel. By matching the reference's dtype
path exactly, we ensure correctness.

## Expected Performance
- Similar to attempt 51 (~88 µs geomean) with possible improvement from inference_mode
- inference_mode disables autograd tracking, potentially speeding up tensor creation
