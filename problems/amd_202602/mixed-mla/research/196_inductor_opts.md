# Attempt 196: Inductor optimization options

## Changes vs 194
- torch.compile with options={"epilogue_fusion": True, "coordinate_descent_tuning": True}
- epilogue_fusion: fuses epilogue (softmax cast, etc.) with GEMM kernel
- coordinate_descent_tuning: enables extra autotuning of generated kernels
