# #109 HIP Graph Capture for quant+GEMM

## Hypothesis
Capture dynamic_mxfp4_quant(A) + gemm_afp4wfp4 into a HIP graph.
Replaying eliminates kernel launch overhead (~2-3µs for 2-3 kernel launches).
Copy overhead: A=256KB max, B_q=7MB max, B_scale=224B max → ~1µs at 5TB/s.
Net savings: ~1-2µs per call for M>16 shapes.

## Approach
1. First call: warmup JIT + capture graph
2. Subsequent calls: copy A/B_q/B_scale into static buffers, replay graph
3. M<=16 KSPLIT=1: keep fused kernel (already single launch, no graph benefit)

## Risks
- graph capture might fail (dynamic_mxfp4_quant may have CPU-GPU sync)
- copy overhead might offset launch savings
- graph capture during benchmark warmup adds latency

## Results
MUCH WORSE: 25-37µs (vs 12-21µs baseline). Graph capture + copy introduce
GPU-CPU sync barriers that serialize all operations.
Even M=4 (fused path, no graph) got 25.9µs due to sync from graph capture
on M=32 shape polluting the GPU pipeline.
HIP graphs on ROCm 7.1 are not suitable for this use case.
