# v65: ROCm Environment Variable Tuning

## Variables
- `HIP_FORCE_DEV_KERNARG=1`: Forces device kernel arguments, reducing kernel launch overhead
- `GPU_MAX_HW_QUEUES=2`: Limits HW queues to 2 (recommended by AMD for vLLM)

## Rationale
From ROCm documentation for MI355X:
- HIP_FORCE_DEV_KERNARG reduces the overhead of passing arguments to GPU kernels
- GPU_MAX_HW_QUEUES=2 reduces queue management overhead

## Expected Impact
- ~5-10µs reduction in per-call Python/HIP dispatch overhead
- Affects ALL kernel launches (sorting, quant, stage1, stage2)
