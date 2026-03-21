# MoE MXFP4 Optimization Ideas

Based on research of AITER library and README hints.

## Current Status
- **Baseline achieves 1.14x geometric mean speedup** over AITER reference
- AITER automatically selects optimal kernel configurations
- Limited user-tunable parameters available
- `doweight_stage1=False` is a correctness requirement (not a tuning knob)

## Optimization Opportunities (from README)

1. **Custom tiling / scheduling**
   - The CK kernel uses a fixed tile strategy
   - For small batch sizes (bs=16) or skewed expert distributions, custom schedule may reduce idle waves

2. **Activation quantization fusion**
   - Reference quantizes activations separately before the GEMM
   - Fusing dynamic MXFP4 quantization into Stage 1 GEMM prologue saves memory round-trip

3. **Inter-stage fusion**
   - Reference runs Stage 1 and Stage 2 as separate kernel launches
   - Fusing both stages eliminates intermediate buffer write/read

4. **Expert-parallel wave scheduling**
   - With 257 experts but only 9 active per token, most expert slots empty
   - Work-stealing or compact-dispatch strategy can minimize wasted wavefronts

5. **Shared expert fusion**
   - Shared expert processes ALL tokens unconditionally (weight=1.0)
   - Could compute as dense GEMM (no routing overhead) and fuse with reduction

6. **Split-K for large M**
   - For bs=512 with E=33, d_expert=2048, GEMMs large enough for split-K parallelism

## AITER fused_moe Parameters to Try

1. **doweight_stage1=True** (currently False)
   - Apply router weights in stage 1 instead of final reduction
   - May have different performance characteristics

2. **expert_mask** - Only for distributed EP, not applicable here

3. **a1_scale / a2_scale** - Activation scales, currently None (uses dynamic quant)

## Alternative AITER Functions

1. **fused_topk + moe_sorting**
   - Two-stage approach with better memory locality
   - Sort tokens by expert before GEMM

2. **asm_moe**
   - Assembly-optimized MoE kernel
   - Potentially 3x speedup

3. **gemm_a4w4**
   - Used directly for MXFP4 matrix multiply
   - Could build custom MoE from primitives

## Strategies to Try

### Attempt 2: doweight_stage1=True
Change `doweight_stage1=False` to `True` and benchmark

### Attempt 3: Manual expert sorting
Pre-sort tokens by expert to improve memory coalescing

### Attempt 4: Shared expert separation
Compute shared expert as dense GEMM separately

### Attempt 5: Custom Triton kernel
Write custom kernel for specific shapes if AITER is suboptimal
