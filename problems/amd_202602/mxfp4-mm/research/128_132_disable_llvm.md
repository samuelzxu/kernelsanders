# #128-132 DISABLE_LLVM_OPT Experiments

## Inspiration
Leaderboard: `submission_v220_disable_lsr_disable_machine_licm.py` at 9.7µs

## Experiments
- #128: DISABLE_LLVM_OPT='disable-lsr,disable-machine-licm' → no effect
- #129: Same + cache clear → no effect
- #130: DISABLE_LLVM_OPT='1' (disable ALL) → 2x SLOWER (confirms var works!)
- #131: Space-separated flags → no effect
- #132: Just 'disable-lsr' + multiple env vars → no effect

## Analysis
DISABLE_LLVM_OPT=1 (boolean) IS read and disables ALL LLVM opts (confirming
the env var works). But specific flag values ('disable-lsr') are NOT applied
to the AMD backend's compilation pipeline.

The AMD backend likely handles LLVM flags differently from NVIDIA backend.
The competitor at 9.7µs is probably:
1. Monkey-patching the Triton AMD backend compiler.py to inject LLVM flags
2. Or passing flags through a different mechanism (LLVM_ARGS, etc.)

## Next steps
Need to find the exact code path in triton AMD backend where LLVM opt level
is set, and inject the -disable-lsr -disable-machine-licm flags there.
