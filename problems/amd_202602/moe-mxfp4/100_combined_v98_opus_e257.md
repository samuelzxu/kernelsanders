# v100: Combined v98 + Opus Sorting + Extended cktile for E=257 bs=512

## Approach

Three improvements stacked:

### 1. All v98 Optimizations (Pre-alloc + NT loads + cktile for sparse)
- `AITER_USE_NT=1` for non-temporal weight loads (all shapes)
- Pre-allocated sorting buffers via `_fast_sorting()` with `moe_sorting_fwd`
- cktile (bf16 activations, skip FP4 quant) for:
  - tok/exp < 5 → sk=2 (E=257 bs=16/128)
  - tok/exp < 40 AND expert<=33 → sk=1 (E=33 bs=16/128)

### 2. Opus Sorting (from v99)
- Uses `moe_sorting_opus_fwd` instead of `moe_sorting_fwd` if available
- Alternative AMD sorting implementation, potentially faster for certain distributions

### 3. Extended cktile for E=257 bs=512 (NEW)
E=257 bs=512 has:
- inter_dim=256 (very small, less quant overhead to save)
- tok/exp ≈ 17.9 (moderate sparse)
- Currently using default CK 2-stage with FP4 quant (244µs)

Adding condition: `tok/exp < 20 AND inter_dim <= 256 → cktile sk=1`
This extends the cktile path to E=257 bs=512 which has the same small inter_dim
as the E=257 shapes that already use cktile (bs=16/128).

## Rationale
The key insight is that cktile's benefit comes from skipping FP4 quant overhead.
For inter_dim=256 shapes, the quant overhead is proportionally smaller but the
BF16 GEMM in cktile should be equally fast (same compute budget).

E=33 bs=128 (tok/exp=34.9) successfully uses cktile sk=1.
E=257 bs=512 (tok/exp=17.9) has FEWER tokens per expert, so cktile sk=1 should also work.

## Expected Impact
- Opus sorting: potentially 1-3% improvement on sorting-dominated shapes
- E=257 bs=512 cktile: if cktile saves even 10µs (same proportional as E=33), geomean improves
- Combined: potentially 2-5% total improvement over v98
