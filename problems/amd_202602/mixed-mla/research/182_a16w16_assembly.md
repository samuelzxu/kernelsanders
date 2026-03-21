# Attempt 182: a16w16 assembly kernel for kv<=1024

## Discovery
aiter has specialized kernels we never tried:
- `mla_dec_stage1_bf16_a16w16_subQ16_mqa16.co` (specialized for NH=16 MQA decode)
- Multiple a16w16 variants with different tiling

## Hypothesis
The a16w16 kernel (bf16 Q + bf16 KV) might be faster per-byte despite 2x
more KV bandwidth because:
- No fp8 dequantization inside the kernel
- Better memory alignment (bf16 = 2-byte aligned)
- Specialized kernel tuned for exact (NH=16, MQA, decode) config
