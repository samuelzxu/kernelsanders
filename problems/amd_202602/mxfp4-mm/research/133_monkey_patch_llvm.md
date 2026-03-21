# #133 Monkey-patch llvm.optimize_module for LSR+LICM disable

## Discovery
The AMD backend calls:
```python
llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3, options.arch, '', [], options.enable_fp_fusion)
```
at line 457 of compiler.py. The 5th argument `[]` is the flags list.

## Approach
Patch `_triton.llvm.optimize_module` to inject `-disable-lsr -disable-machine-licm`
into the flags list. Combined with cache clear to force recompilation.

## Risk
- The C extension function might not accept custom flags
- The flags might have different names in the LLVM version used
- The monkey-patch might fail on the runner's Triton version
