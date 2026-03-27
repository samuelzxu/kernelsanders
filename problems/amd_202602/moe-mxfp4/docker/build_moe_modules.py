"""
Build AITER modules for MOE-MXFP4 via cross-compilation (no GPU required).
Forces gfx950 target via GPU_ARCHS environment variable.

Modules needed:
1. module_aiter_enum          - basic enums
2. module_moe_sorting_opus    - opus sorting kernel
3. module_moe_sorting         - standard sorting kernel
4. module_quant               - quantization kernels
5. module_moe_cktile2stages   - cktile BF16 MOE kernels
6. module_moe_ck2stages_fp4x2_... - CK FP4 MOE kernels
7. module_activation          - activation kernels
"""
import os
import sys
import time

# CRITICAL: Set GPU_ARCHS before any aiter imports
# This tells AITER's JIT system to target gfx950 without a physical GPU
os.environ["GPU_ARCHS"] = "gfx950"
os.environ["CU_NUM"] = "256"  # MI355X has 256 CUs
os.environ["HIP_VISIBLE_DEVICES"] = ""  # No GPU needed
os.environ["AITER_BUILD_VERBOSE"] = "1"

def build_modules():
    start = time.time()

    print("=== Starting AITER module builds for gfx950 ===", flush=True)
    print(f"GPU_ARCHS={os.environ.get('GPU_ARCHS')}", flush=True)
    print(f"CU_NUM={os.environ.get('CU_NUM')}", flush=True)

    # Step 1: Import aiter (builds module_aiter_enum)
    try:
        print("\n[1/7] Building module_aiter_enum...", flush=True)
        t0 = time.time()
        import aiter
        print(f"  Done in {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        print(f"  Failed: {e}", flush=True)
        import traceback; traceback.print_exc()
        return

    # Step 2: Build sorting modules by importing the functions
    # These are registered as custom ops and built on first import
    try:
        print("\n[2/7] Building module_moe_sorting_opus...", flush=True)
        t0 = time.time()
        # Force the JIT build by accessing the function
        if hasattr(aiter, 'moe_sorting_opus_fwd'):
            print(f"  Already available in {time.time()-t0:.1f}s", flush=True)
        else:
            print("  moe_sorting_opus_fwd not found in aiter", flush=True)
    except Exception as e:
        print(f"  Failed: {e}", flush=True)

    try:
        print("\n[3/7] Building module_moe_sorting...", flush=True)
        t0 = time.time()
        if hasattr(aiter, 'moe_sorting_fwd'):
            print(f"  Already available in {time.time()-t0:.1f}s", flush=True)
        else:
            print("  moe_sorting_fwd not found in aiter", flush=True)
    except Exception as e:
        print(f"  Failed: {e}", flush=True)

    # Step 3: Build quant module
    try:
        print("\n[4/7] Building module_quant...", flush=True)
        t0 = time.time()
        from aiter.ops.quant import dynamic_mxfp4_quant
        print(f"  Done in {time.time()-t0:.1f}s", flush=True)
    except ImportError:
        try:
            from aiter import dynamic_mxfp4_quant
            print(f"  Done in {time.time()-t0:.1f}s", flush=True)
        except ImportError as e:
            print(f"  Failed: {e}", flush=True)
            # Try to find it
            import aiter
            quant_attrs = [a for a in dir(aiter) if 'quant' in a.lower()]
            print(f"  Available quant attrs: {quant_attrs}", flush=True)

    # Step 4: Build activation module
    try:
        print("\n[5/7] Building module_activation...", flush=True)
        t0 = time.time()
        from aiter import silu_and_mul
        print(f"  Done in {time.time()-t0:.1f}s", flush=True)
    except ImportError:
        try:
            import aiter
            act_attrs = [a for a in dir(aiter) if 'silu' in a.lower() or 'activation' in a.lower()]
            print(f"  Available activation attrs: {act_attrs}", flush=True)
        except Exception as e2:
            print(f"  Failed: {e2}", flush=True)

    # Step 5: Build CK MOE modules
    # These require calling the actual kernel functions, which triggers JIT compilation
    try:
        print("\n[6/7] Building module_moe_cktile2stages...", flush=True)
        t0 = time.time()
        from aiter.fused_moe import cktile_moe_stage1, cktile_moe_stage2
        print(f"  Imported in {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        print(f"  Failed: {e}", flush=True)
        import traceback; traceback.print_exc()

    try:
        print("\n[7/7] Building module_moe_ck2stages (FP4)...", flush=True)
        t0 = time.time()
        from aiter.fused_moe import ck_moe_stage1, ck_moe_stage2
        print(f"  Imported in {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        print(f"  Failed: {e}", flush=True)
        import traceback; traceback.print_exc()

    # Step 6: Try to force-build the JIT modules by calling the build system directly
    print("\n=== Attempting direct JIT builds ===", flush=True)
    try:
        # Import the JIT core and build modules directly
        sys.path.insert(0, "/home/runner/aiter/aiter/jit")
        sys.path.insert(0, "/home/runner/aiter/aiter/jit/utils")
        from core import build_module, get_module

        # List of modules to build
        modules_to_build = [
            "module_moe_sorting_opus",
            "module_moe_sorting",
            "module_quant",
            "module_activation",
            "module_moe_cktile2stages",
        ]

        for mod_name in modules_to_build:
            try:
                t0 = time.time()
                print(f"\nBuilding {mod_name}...", flush=True)
                mod = get_module(mod_name)
                if mod is not None:
                    print(f"  Got module in {time.time()-t0:.1f}s", flush=True)
                else:
                    print(f"  Module returned None", flush=True)
            except Exception as e:
                print(f"  Failed: {e}", flush=True)
    except Exception as e:
        print(f"Direct JIT build failed: {e}", flush=True)

    elapsed = time.time() - start
    print(f"\n=== Build complete in {elapsed:.1f}s ===", flush=True)

    # List built modules
    jit_dir = "/home/runner/aiter/aiter/jit"
    print(f"\nBuilt modules in {jit_dir}:", flush=True)
    if os.path.isdir(jit_dir):
        for f in sorted(os.listdir(jit_dir)):
            if f.endswith(".so"):
                size = os.path.getsize(os.path.join(jit_dir, f))
                print(f"  {f}  ({size:,} bytes)", flush=True)
    else:
        print(f"  Directory not found!", flush=True)

if __name__ == "__main__":
    build_modules()
