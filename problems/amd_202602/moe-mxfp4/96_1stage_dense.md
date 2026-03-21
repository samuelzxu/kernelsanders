# v96: 1-Stage Fused Kernel for Dense Shapes

## Approach
Uses `aiter.fmoe_g1u1` for dense shapes (E=33 bs=512 d=512, E=33 bs=512 d=2048,
E=257 bs=512 d=256). This pre-compiled kernel fuses:
- Input FP4 quantization
- GEMM1 (gate_up projection + SwiGLU activation)
- GEMM2 (down projection + weighted reduction)
into a SINGLE kernel launch.

### Why this could help
2-stage path for dense shapes has 5 kernel launches:
1. moe_sorting
2. fused_quant_sort (input FP4 quant + scale sort)
3. CK stage1 (GEMM1)
4. fused_quant_sort (intermediate FP4 requant + scale sort)
5. CK stage2 (GEMM2)

1-stage path eliminates intermediate requant entirely:
1. moe_sorting
2. FP4 quant of input
3. moe_mxfp4_sort of scales
4. fmoe_g1u1 (fused GEMM1+GEMM2)

### Key details
- Uses RAW (un-shuffled) weights for 1-stage path
- Uses SHUFFLED weights for 2-stage cktile path (sparse shapes)
- Excludes E=65 from 1-stage (known precision issue with 1-2 element tolerance violations)
- Pre-allocated sorting buffers (from v95)

## Risk
- 1-stage kernel may be slower for large inter_dim (d=2048) due to register pressure
- Precision might fail for some shapes not tested before
- RAW vs shuffled weight format matters

## Changes from v85
- Added 1-stage path with `fmoe_g1u1` for dense shapes
- Added pre-allocated sorting buffers
- Custom_kernel now branches on metadata.run_1stage
