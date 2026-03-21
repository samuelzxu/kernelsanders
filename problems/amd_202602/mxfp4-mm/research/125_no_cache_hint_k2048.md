# #125 cache_modifier=None for K=2048
M=64: 20.8µs (worse, +1.0µs). The ".cg" L2 cache hint IS beneficial for
B data even in ranked benchmark. It improves spatial locality of B reads
within the GEMM kernel execution, regardless of cross-iteration caching.
