# Attempt 181: HIP quantization for Q fp8

## Change
Replace Triton `dynamic_per_tensor_quant_fp8_i8` with HIP `aiter.dynamic_per_tensor_quant`.
- Pre-compiled HIP kernel (no Triton dispatch overhead)
- Uses torch.empty for scale (saves zero-fill kernel, ~0.5µs)
- Also removes the `from aiter.ops.triton.quant.quant import ...` which may trigger Triton init
