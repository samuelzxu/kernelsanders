# Attempt 242: Fused MXFP4 Decode Attention Kernel

## Architecture
- Stage 1: Custom Triton kernel with tl.dot_scaled for FP4×FP4 Q@K^T
  - Q bf16 → quantized to MXFP4 via _mxfp4_quant_op in registers
  - K loaded from kv_data["mxfp4"] (fp4x2 packed, 288 bytes/token)
  - V loaded from kv_data["bf16"] (512 bytes/token) — safe, no dequant complexity
  - Online softmax with split-K parallelization
- Stage 2: Triton reduction kernel (log-sum-exp across splits)

## Expected Performance
- K bandwidth: 306 bytes/token (vs 576 for fp8) = 47% savings
- V bandwidth: 1024 bytes/token (bf16, unchanged for correctness)
- Total: ~1330 bytes/token — only slightly better than fp8's 576 (fused single pass)
- BUT: eliminates metadata overhead, simpler kernel dispatch
- NOTE: for maximum speedup, V should also come from MXFP4 (future optimization)

## Risks
- tl.dot_scaled with DQ=576 might not work (non-standard head dim)
- _mxfp4_quant_op register pressure with BLOCK_H=16, DQ=576
- JIT compilation timeout (kernel is complex)
- Correctness: MXFP4 quantization noise within tolerance?
