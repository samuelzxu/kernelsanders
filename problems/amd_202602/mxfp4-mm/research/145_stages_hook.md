# #145 Stages Inspection Hook

## Success: Hook mechanism works!
`knobs.runtime.add_stages_inspection_hook` successfully intercepts ALL kernel
compilations (10 per run). The hook wraps the llir stage and sets
DISABLE_LLVM_OPT=disable-lsr before each compilation.

## Failure: Same underlying issue
The DISABLE_LLVM_OPT env var with specific flags doesn't affect the AMD
backend's optimize_module C binding. Same results as #118.

## Key learning
The `add_stages_inspection_hook` is the CORRECT mechanism for modifying
compilation behavior. We can:
- Replace compilation stages entirely
- Inject env vars per-compilation
- Modify options before compilation
But the LLVM pass control requires modifying the C binding itself.
