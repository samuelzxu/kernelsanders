# Future Optimization Ideas

## Not Yet Tried

### 1. MXFP4 with Fused Dequantization
- **Goal**: 4x bandwidth savings over bf16, 2x over fp8
- **Challenge**: aiter mla_decode_fwd doesn't support MXFP4 natively
- **Approach**: Write custom Triton kernel that:
  - Loads MXFP4 KV tiles from HBM
  - Dequantizes in registers/LDS to bf16
  - Computes QK^T and softmax·V without writing back
- **Status**: Complex, requires significant Triton expertise

### 2. Metadata Caching
- **Goal**: Reduce overhead of metadata creation
- **Approach**: Cache work buffers for repeated calls with same batch_size
- **Risk**: May not help if shapes vary between calls

### 3. Pre-allocated kv_indices
- **Goal**: Avoid torch.arange allocation on each call
- **Approach**: Pre-allocate and reuse kv_indices tensor
- **Expected Impact**: Minor

### 4. bf16 Q + fp8 KV (a16w8)
- **Goal**: Skip Q quantization overhead
- **Trade-off**: a16w8 is slower than a8w8 according to docs
- **Verdict**: Probably not worth it

### 5. CUDA Graphs
- **Goal**: Reduce kernel launch overhead
- **Challenge**: May not work well with dynamic shapes
- **Status**: Need to investigate aiter support

### 6. Different fast_mode Settings
- **Goal**: Potentially faster metadata computation
- **Risk**: May affect accuracy or correctness
- **Status**: Need to test

## Web Research Findings

### From AMD ROCm Blogs:
- AITER MLA offers 17x performance boost for decode
- MI355X has native FP4/FP6 support
- Petit library enables FP16/BF16 × FP4 mixed precision

### From Triton/vLLM:
- Triton attention on MI300 achieves ~5.8x speedup
- Same Triton source works on both NVIDIA and AMD
- MXFP4 used for MoE weights, attention uses bf16 in GPT-OSS

## Priority Ranking
1. MXFP4 fused dequant (high effort, high reward)
2. Metadata caching (low effort, low reward)
3. Different num_kv_splits tuning (done)
4. fast_mode experimentation (low effort, uncertain reward)
