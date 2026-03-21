# #89 Fused A16WFP4 for All Shapes

## Hypothesis
Use `gemm_a16wfp4` (fused bf16→fp4 quant + GEMM) for M>=32 shapes.
This eliminates the separate `dynamic_mxfp4_quant(A)` call, saving ~2-3µs.
The A quantization happens inside the GEMM kernel register file.

## Changes
- M<=16: Keep custom fused kernel (already fused)
- M>=32: Use `gemm_a16wfp4(A, B_q_uint8, B_scale)` instead of quant+gemm_afp4wfp4
- Inject A16WFP4 configs matching our benchmark shapes (K in filename = 2*K)
- Still inject AFP4WFP4 configs for the M<=16 path

## Config Key Mapping (A16WFP4 uses 2*K in filenames)
- N=2880, K=512 → N=2880-K=1024
- N=4096, K=512 → N=4096-K=1024
- N=2112, K=7168 → N=2112-K=14336
- N=7168, K=2048 → N=7168-K=4096
- N=3072, K=1536 → N=3072-K=3072

## Results
TBD
