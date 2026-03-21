# v93: cktile stage1 + flydsl stage2 for Dense Shapes

## Key Discovery
flydsl IS available on the runner (v92 confirmed).
flydsl stage2 supports a_dtype="fp16" (bf16 activations + fp4 weights).
This allows bf16 intermediate → flydsl stage2, skipping requant.

## Approach for E=33 bs=512 d=512 (139 tok/exp)
- cktile stage1 (bf16, no input quant): ~44µs (10% slower than CK FP4)
- Skip requant: saves ~15µs
- flydsl stage2 (bf16 activations + fp4 weights): hopefully faster than CK stage2
- Net: potentially saves ~11µs → ~200µs (vs 211µs baseline)
