# Flash Attention Decode Optimization Research

## Key Insights from Literature

### Decode Bottlenecks
- Query sequence length = 1 means parallelism is limited
- Main bottleneck: loading KV cache as fast as possible
- Memory bandwidth bound, not compute bound

### Split-K / Flash-Decoding
- Split attention along KV sequence length
- Number of splits determined by heuristic at launch
- Separate post-processing reduction kernel to combine results
- This is what `num_kv_splits` controls in aiter

### FlashInfer Optimizations
- Dynamic scheduler for load balancing
- Versatile tile size selection (critical for decode)
- StreamK-like optimizations for variable length sequences
- Achieves ~100% GPU bandwidth utilization for long sequences

### LeanAttention
- Uses Stream-K style load balancing
- Targets poor GPU occupancy in decode phase
- Achieves nearly peak occupancy

### Memory Hierarchy
- HBM: 40-80GB, ~1.5-2.0 TB/s bandwidth (slowest)
- SRAM: ~15x faster than HBM
- Fused kernels avoid materializing N² attention matrix

## Application to Our Problem

### Current State
- Using aiter's Split-K with num_kv_splits tuning
- Already beating reference by 1.6-2.1x on small batches

### Potential Improvements
1. **Better num_kv_splits heuristic**: Current is simple, could profile more
2. **Tile size tuning**: aiter may have internal tile size params
3. **MXFP4 for bandwidth**: 4x less data to load from HBM

## Sources
- [FlashAttention-3 NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/7ede97c3e082c6df10a8d6103a2eebd2-Paper-Conference.pdf)
- [FlashInfer MLSys 2025](https://proceedings.mlsys.org/paper_files/paper/2025/file/dbf02b21d77409a2db30e56866a8ab3a-Paper-Conference.pdf)
- [FlashInfer Blog](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html)
