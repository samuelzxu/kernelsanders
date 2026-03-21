# #146-147 Triton Version + Optimization Level

## #146 Findings
- Triton version: 3.6.0
- optimize_module signature: `(mod, opt, arch='', features='', flags=Sequence[str]=[], enable_fp_fusion=False)`
- flags IS Sequence[str] → accepts string list (confirmed)
- enable_fp_fusion defaults to False
- pip install blocked (externally-managed-environment)
- Available opt levels: O0, O1, O2, O3, Os, Oz

## #147 OPTIMIZE_Os
Patch confirmed applied. Same results as O2 and O3.
O2 = O3 = Os for FP4 GEMM → LLVM optimization level is irrelevant.
The dot_scaled MFMA lowering happens at MLIR level, and LLVM just assembles it.
The generated assembly is identical regardless of optimization level.

## Complete opt level results:
| Level | Effect |
|-------|--------|
| O0 | 2x slower (DISABLE_LLVM_OPT=1) |
| O1 | Not tested |
| O2 | Same as O3 |
| O3 | Baseline (#118) |
| Os | Same as O3 |
| Oz | Not tested |
