# v53: Parallel Background Pre-build

## Hypothesis
Run module_moe_asm build in a BACKGROUND subprocess thread while the
main process runs tests (which use 2-stage CK). By the time benchmarks
start, the .so file is cached. The build runs concurrently, not blocking
the test phase.

## Key Innovation
- `threading.Thread(target=_background_prebuild, daemon=True)` starts build
  at import time
- The subprocess uses a SEPARATE Python process to trigger the JIT build
- The main process continues with tests using 2-stage CK (already cached)
- On first benchmark call, `_prebuild_thread.join(timeout=60)` waits
  for the build to finish
- After build, 1-stage kernel is available via cached .so file

## Expected Timeline
- t=0: Import → start background prebuild thread + subprocess
- t=0-30s: Test phase runs with 2-stage CK (tests pass)
- t=30s: Background subprocess finishes building module_moe_asm.so
- t=30s: Benchmark phase starts, first E=33 call joins thread (instant)
- t=30-540s: Benchmark runs with 1-stage for E=33, 2-stage for E=257
