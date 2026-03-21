# #141 LLVM O2 vs O3

Patch confirmed applied (stderr: `[PATCH] Changed LLVM opt O3→O2`).
Same results as O3. LLVM O2 and O3 produce equivalent code for FP4 GEMM.
The harmful passes (LSR, LICM) run at both O2 and O3 levels.
Disabling specific passes requires modifying the C binding's pass manager,
which is not exposed through the Python API.
