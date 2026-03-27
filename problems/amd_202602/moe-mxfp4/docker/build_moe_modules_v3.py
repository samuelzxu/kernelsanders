"""
Build AITER JIT modules by directly invoking the build system.
Bypasses the need for GPU tensors by calling build_module directly.
"""
import os
import sys
import time

# Force gfx950 target
os.environ["GPU_ARCHS"] = "gfx950"
os.environ["CU_NUM"] = "256"

print(f"Python: {sys.version}", flush=True)
print(f"GPU_ARCHS: {os.environ.get('GPU_ARCHS')}", flush=True)

start = time.time()

# Add AITER JIT paths
AITER_DIR = "/home/runner/aiter"
sys.path.insert(0, f"{AITER_DIR}/aiter/jit")
sys.path.insert(0, f"{AITER_DIR}/aiter/jit/utils")

# Import aiter first to set up the module
import aiter

# Import JIT build functions
from core import build_module, get_args_of_build, get_module

# Modules to build
MODULES = [
    "module_moe_sorting_opus",
    "module_moe_sorting",
    "module_quant",
    "module_activation",
    "module_moe_cktile2stages",
    # CK FP4 2-stage module has a dynamic name based on configs
    # We'll try to build it with the common configuration
]

for md_name in MODULES:
    print(f"\n=== Building {md_name} ===", flush=True)
    t0 = time.time()
    try:
        # Check if already built
        try:
            mod = get_module(md_name)
            print(f"  Already exists, skipping ({time.time()-t0:.1f}s)", flush=True)
            continue
        except (ModuleNotFoundError, Exception):
            pass

        # Get build args from optCompilerConfig.json
        d_args = get_args_of_build(md_name)

        # Extract build parameters
        srcs = d_args["srcs"]
        flags_extra_cc = d_args["flags_extra_cc"]
        flags_extra_hip = d_args["flags_extra_hip"]
        blob_gen_cmd = d_args["blob_gen_cmd"]
        extra_include = d_args["extra_include"]
        extra_ldflags = d_args["extra_ldflags"]
        verbose = d_args["verbose"]
        is_python_module = d_args["is_python_module"]
        is_standalone = d_args["is_standalone"]
        torch_exclude = d_args["torch_exclude"]
        third_party = d_args.get("third_party", [])
        hipify = d_args.get("hipify", False)

        print(f"  Sources: {len(srcs)} files", flush=True)
        print(f"  Third-party: {third_party}", flush=True)

        # Build the module
        build_module(
            md_name,
            srcs,
            flags_extra_cc,
            flags_extra_hip,
            blob_gen_cmd,
            extra_include,
            extra_ldflags,
            verbose,
            is_python_module,
            is_standalone,
            torch_exclude,
            third_party,
            hipify=hipify,
        )
        print(f"  Done in {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)
        import traceback; traceback.print_exc()

# Now try the CK FP4 2-stage module
# This module name is dynamically generated based on quantization config
CK_MOD_NAME = "module_moe_ck2stages_fp4x2_fp4x2_preshuffle_on_b16_silu_per_1x32_mulWeightStage2_"
print(f"\n=== Building {CK_MOD_NAME} ===", flush=True)
t0 = time.time()
try:
    # This module requires special gen_func to generate build args
    # It's triggered by ck_moe_stage1/stage2 calls with specific params
    # Let's try to get its build args
    try:
        d_args = get_args_of_build(CK_MOD_NAME)
        srcs = d_args["srcs"]
        build_module(
            CK_MOD_NAME,
            srcs,
            d_args["flags_extra_cc"],
            d_args["flags_extra_hip"],
            d_args["blob_gen_cmd"],
            d_args["extra_include"],
            d_args["extra_ldflags"],
            d_args["verbose"],
            d_args["is_python_module"],
            d_args["is_standalone"],
            d_args["torch_exclude"],
            d_args.get("third_party", []),
            hipify=d_args.get("hipify", False),
        )
        print(f"  Done in {time.time()-t0:.1f}s", flush=True)
    except KeyError:
        print(f"  Module not in optCompilerConfig.json, trying gen_func path...", flush=True)
        # The CK 2-stage module is generated dynamically
        # We need to import the gen_func and call it to get build args
        try:
            from aiter.ops.ck_moe_2stages_op import gen_ck_moe_2stages_build_args
            # Try to generate build args for our configuration
            build_args = gen_ck_moe_2stages_build_args(
                q_dtype_a="fp4x2",
                q_dtype_w="fp4x2",
                preshuffle="on",
                dtype="b16",
                activation="silu",
                quant_type="per_1x32",
                doweight_stage1=False,
            )
            build_module(
                CK_MOD_NAME,
                build_args["srcs"],
                build_args["flags_extra_cc"],
                build_args["flags_extra_hip"],
                build_args["blob_gen_cmd"],
                build_args["extra_include"],
                build_args["extra_ldflags"],
                build_args["verbose"],
                build_args["is_python_module"],
                build_args["is_standalone"],
                build_args["torch_exclude"],
                build_args.get("third_party", []),
                hipify=build_args.get("hipify", False),
            )
            print(f"  Done in {time.time()-t0:.1f}s", flush=True)
        except Exception as e2:
            print(f"  Gen func approach also failed: {e2}", flush=True)
            import traceback; traceback.print_exc()
except Exception as e:
    print(f"  FAILED: {e}", flush=True)
    import traceback; traceback.print_exc()

# List results
elapsed = time.time() - start
print(f"\n=== Build complete in {elapsed:.1f}s ===", flush=True)

jit_dir = "/home/runner/aiter/aiter/jit"
print(f"\nBuilt modules in {jit_dir}:", flush=True)
for f in sorted(os.listdir(jit_dir)):
    if f.endswith(".so"):
        size = os.path.getsize(os.path.join(jit_dir, f))
        print(f"  {f}  ({size:,} bytes)", flush=True)
