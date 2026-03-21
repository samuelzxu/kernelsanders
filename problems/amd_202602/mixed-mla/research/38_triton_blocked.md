# Attempt 38: Custom Triton Kernel - BLOCKED

## Finding
The benchmark platform rejects any code that launches GPU work on a non-default
CUDA stream. Triton kernels use their own stream, causing:

```
Server returned status 500: Your code contains work on another stream.
```

## Implications
- Custom Triton kernels cannot be used
- MXFP4 fused dequant kernel is impossible
- We are limited to aiter's pre-built assembly kernels
- Only the following kernel variants are available:
  - mla_a16w16 (bf16 Q + bf16 KV) - persistent and non-persistent
  - mla_a16w8 (bf16 Q + fp8 KV) - persistent only
  - mla_a8w8 (fp8 Q + fp8 KV) - persistent and non-persistent
