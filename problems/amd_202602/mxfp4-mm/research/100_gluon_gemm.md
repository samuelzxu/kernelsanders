# #100 Gluon Variant gemm_afp4wfp4

## Hypothesis
The Gluon variant uses:
- @gluon.jit decorator with explicit MFMA operations
- Built-in remap_xcd for XCD distribution
- matrix_instr_nonkdim=32 (32x32 MFMA tiles vs 16x16)
- Explicit memory layout control

Same API as basic variant. Inject per-shape configs in gluon/ directory.
If Gluon import fails, fall back to basic variant.

## Configs
Used matrix_instr_nonkdim=32 for all shapes (Gluon default).
Matched tile sizes to our benchmark shapes for CU occupancy.

## Results
TBD
