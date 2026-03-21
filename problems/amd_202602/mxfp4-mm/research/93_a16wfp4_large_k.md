# #93 gemm_a16wfp4 for KSPLIT>1 shapes

## Hypothesis
For shapes with KSPLIT>1 (K=7168/M=16, K=1536/M=256):
- Current path: dynamic_mxfp4_quant(A) + gemm_afp4wfp4 (2 kernel launches)
- New path: gemm_a16wfp4 (1 kernel launch, quant fused inside)
- Saves 1 kernel launch (~1-2µs) for the quant step

## Shapes affected
- M=16, N=2112, K=7168 (KSPLIT=8): Currently 21.1µs
- M=256, N=3072, K=1536 (KSPLIT=2): Currently 20.6µs
- M=64, N=7168, K=2048 (KSPLIT=1): NOT affected, stays on afp4wfp4
- M=4/32, K=512 (KSPLIT=1): NOT affected

## A16WFP4 config injection
Injected matching configs for N=2112-K=7168 and N=3072-K=1536
using same parameters as AFP4WFP4 configs.

## Results
TBD
