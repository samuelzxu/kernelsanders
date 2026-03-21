# #142 Force fp_fusion=True
Patch confirmed applied. Same results - was already True by default.

## Complete LLVM backend patching summary (128-142):
Every parameter to `llvm.optimize_module(mod, opt_level, arch, features, flags, fp_fusion)`
has been tested via the copy+redirect mechanism:
- opt_level: O2=O3 (no difference)
- features: '' (can't inject pass control here)
- flags: [] (C binding ignores pass names)
- fp_fusion: True (already default)
Only DISABLE_LLVM_OPT=1 (skip optimize_module entirely) has an effect (2x worse).

The top competitors' speed advantage must come from a mechanism NOT exposed
through the `optimize_module` Python API - likely a modified Triton C binding
or a completely different kernel implementation (custom HIP/MFMA).
