# v71: ksplit=2 cktile Path with BF16 Activations

## Approach
Force `AITER_KSPLIT=2` to activate the cktile path:
- cktile_moe_stage1: takes bf16 activations + fp4 weights, split_k=2
- cktile_moe_stage2: takes bf16 intermediate + fp4 weights

## Kernel Launch Reduction
- Standard: sorting → quant → stage1 → requant → stage2 = 5 launches
- cktile:   sorting → stage1(bf16) → stage2(bf16) = 3 launches
- Savings: ~10µs quant overhead + ~6µs launch overhead = ~16µs

## Risk
- v25 showed ksplit=2 was 2x slower (687µs vs 340µs for d=2048)
- But v25 didn't have HIP_FORCE_DEV_KERNARG=1 and block_m=16 for cktile
- The cktile tiles may not be optimal for our shapes
- cktile module might need JIT build (separate from CK 2-stage)
