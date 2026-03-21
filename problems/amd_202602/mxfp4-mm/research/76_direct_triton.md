# #76 Direct Triton Kernel Call

## Hypothesis
The `gemm_afp4wfp4` Python wrapper adds overhead:
- `torch_compile_guard` decorator
- Config serialization/deserialization
- Logger.info() calls
- `arch_info.is_fp4_avail()` assertion
- Multiple dict lookups through the call stack

By calling `_gemm_afp4wfp4_kernel[grid](...)` directly, we skip all wrapper layers.

## Expected savings: 1-3µs per call for M≥32 shapes

## Implementation
- Import the raw Triton kernel and config functions
- Inline the config logic (get_config → get_splitk → grid → launch)
- Skip: serialize/deserialize, logging, assertions, torch_compile_guard
