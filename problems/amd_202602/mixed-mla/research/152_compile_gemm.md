# Attempt 152: torch.compile for GEMM path

## Hypothesis
torch.compile with Inductor/CK backend can fuse baddbmm+softmax+bmm into
fewer kernel launches, reducing the ~15µs fixed overhead in the GEMM path.

Key: torch.compile uses CK (Composable Kernel) backend on MI355X, NOT Triton.
This avoids the JIT timeout issue that blocked custom Triton kernels.

## Changes vs 137
- GEMM attention function wrapped in torch.compile(dynamic=True)
- Uses baddbmm (not pre-scaled bmm) to keep the fused alpha scaling
- Compilation happens during warmup iterations

## Risk
- Compilation might timeout (900s test_timeout)
- torch.compile might trigger Triton fallback → JIT timeout
- Correctness might differ slightly
- Previous attempt 122 with torch.compile caused correctness failure
  (but that was for the whole function, not just the GEMM part)
