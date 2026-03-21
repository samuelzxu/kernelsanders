# #87 Custom MFMA FP4 GEMM Kernel

## Goal
Write a minimal custom HIP kernel using __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4
for FP4 x FP4 GEMM with E8M0 block scaling.

## MFMA Instruction Details
- 32x32 output tile, 64 K elements per instruction
- FP4 inputs: 64 elements per lane = 32 bytes = 8 uint32 registers
- FP32 accumulator: 16 floats per lane
- E8M0 scale: uint32 packed 4 scale values per lane
- Type codes: Atype=4, Btype=4 (FP4 E2M1)

## Minimal Kernel Plan
- Block: 64 threads (1 wave) for simplicity
- Output tile: 32x32 (one MFMA)
- K loop: iterate in steps of 64 (one MFMA per step)
- A: load from global to registers (no LDS for first version)
- B: load from global to registers
- Scales: load from global to registers
- Output: write bf16 to global

## Lane Mapping (32x32 MFMA)
- 64 threads, each produces 16 FP32 outputs
- lane [0-15]: row within 16-row group
- lane [16-31], [32-47], [48-63]: different column groups
- Each lane reads 64 FP4 elements (32 bytes) of A and B

## Key challenge
- FP4 data packing: 2 FP4 values per byte
- Scale addressing: 1 scale per 32 elements along K
- Need to quantize A (bf16) on-the-fly if doing fused kernel
