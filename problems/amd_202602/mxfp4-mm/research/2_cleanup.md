# Attempt 2: Code Cleanup and Module-level Imports

## Hypothesis
The baseline has unnecessary overhead:
1. Imports inside function (executed on every call)
2. `B.contiguous()` called but B never used (we use B_shuffle)
3. Inner function `_quant_mxfp4` being defined on every call

Moving imports to module level and removing unused code should reduce overhead.

## Changes
- Move all imports to module level
- Remove unused `A.contiguous()` and `B.contiguous()`
- Remove inner function definition, inline the logic
- Streamline the code path

## Result (Ranked Benchmark)
| M   | N    | K    | Time [µs] | vs Baseline |
|-----|------|------|-----------|-------------|
| 4   | 2880 | 512  | 20.7      | -0.2µs      |
| 16  | 2112 | 7168 | 34.7      | +0.1µs      |
| 32  | 4096 | 512  | 22.2      | -0.5µs      |
| 32  | 2880 | 512  | 22.1      | +0.1µs      |
| 64  | 7168 | 2048 | 24.5      | -0.2µs      |
| 256 | 3072 | 1536 | 23.3      | +0.1µs      |

**Conclusion**: No meaningful improvement - the overhead is in the untuned CK kernel, not Python.
Still seeing "not found tuned config in CKGEMM or asmGEMM, will use default config!" for all shapes.
