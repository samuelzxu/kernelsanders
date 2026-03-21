# #140 Disable Debug Line Info + Triton Knobs
TRITON_DISABLE_LINE_INFO=1, MLIR_ENABLE_REMARK_ALL=0, TRITON_ALWAYS_COMPILE=1.
No effect on any shape. These env vars aren't read by the AMD backend.
