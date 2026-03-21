# Attempt 188: No torch.compile — test if it's causing assembly regression

## Hypothesis
torch.compile adds ~5µs to GEMM configs but may cost ~3-5µs on each
of the 5 assembly configs (from Triton init / memory pool pollution).
Net effect of removing compile:
- GEMM: +5µs (lose compilation benefit) × 3 configs
- Assembly: -3-5µs (cleaner memory state) × 5 configs
- Geomean weighted: assembly improvement might dominate
