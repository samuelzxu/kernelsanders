# DeepSeek MLA Research

## Key Findings

### FlashMLA Performance (NVIDIA H800)
- Up to 660 TFlops in compute-bound workloads
- Up to 3000 GB/s memory bandwidth in memory-bound config
- 5-15% improvement over previous releases
- B200 achieves up to 1450 TFlops

### Weight/Matrix Absorption
- Algebraic manipulation of MLA formula
- More efficient implementation
- Requires specialized optimized code to see gains

### Memory Benefits
- ~60x less KV cache than MHA
- ~12x less than GQA
- Token capacity: 54,560 → 512,000 tokens
- Batch size: 13 → 128

### Sparse Attention (DeepSeek-V3.2)
- Token-level sparse attention kernels
- 640 TFlops prefilling
- 410 TFlops decoding

## Implications for Our Work

### Current vs Optimal
- Our kernel: ~100 µs geometric mean
- Reference bs=4/kv=8k: ~113 µs, ours: ~65 µs = 1.7x faster

### Bottlenecks
1. Memory bandwidth (loading KV cache)
2. Reduction overhead from Split-K
3. Q quantization overhead (minor)

### Potential Optimizations
1. MXFP4 KV cache (4x bandwidth savings)
2. Better Split-K heuristics
3. Fused operations

## Sources
- [FlashMLA GitHub](https://github.com/deepseek-ai/FlashMLA)
- [DeepSeek MLA Blog](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html)
- [vLLM DeepSeek Optimization](https://www.redhat.com/en/blog/enhancing-deepseek-models-mla-and-fp8-optimizations-vllm)
