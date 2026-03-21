# #144 Triton Knobs Exploration

## Available knobs dumped from runner:
### compilation:
- override, dump_ir, dump_ir_extract_di_local_variables, store_binary_only
- always_compile, use_ir_loc, enable_asan, disable_line_info
- front_end_debugging, allow_non_constexpr_globals, instrumentation_mode
- ALL default to False/None

### runtime:
- add_stages_inspection_hook: None ← KEY: can replace compilation stages!
- jit_cache_hook, jit_post_compile_hook
- kernel_load_start_hook, kernel_load_end_hook
- launch_enter_hook, launch_exit_hook
- override_arch, debug, interpret

## Key discovery: add_stages_inspection_hook
From AMD backend code:
```python
if knobs.runtime.add_stages_inspection_hook is not None:
    knobs.runtime.add_stages_inspection_hook(self, stages, options, language, None)
```
This hook receives the `stages` dict which maps stage names to functions.
The `llir` stage is: `stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)`
By setting this hook, we could REPLACE the llir stage with a patched version!

## Next step
Set `triton.knobs.runtime.add_stages_inspection_hook` to a function that
patches `stages["llir"]` to modify LLVM optimization.
