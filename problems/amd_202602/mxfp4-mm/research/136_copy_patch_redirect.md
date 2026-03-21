# #136 Copy+Patch+Redirect AMD Backend

## Result
Patch was SUCCESSFULLY APPLIED (confirmed in stderr).
But performance unchanged (~16µs geomean, same as #118).

The `[]` 5th argument to `llvm.optimize_module` is NOT for LLVM pass flags.
The C extension likely ignores/doesn't parse these as pass names.

## Conclusion
The `optimize_module` function signature:
```python
llvm.optimize_module(llvm_mod, opt_level, arch, features, flags, fp_fusion)
```
The `flags` parameter may be for target feature flags, not LLVM pass control.
Disabling specific LLVM passes requires a different mechanism - possibly:
1. Modifying the C++ binding in libtriton.so
2. Using LLVM's pass management API directly
3. The competitor may have a forked Triton with the passes removed
