"""
Build AITER JIT modules for MOE-MXFP4 inside the exact runner image.
Forces gfx950 via GPU_ARCHS env var.

The JIT system builds modules when their functions are first called.
We need to create minimal dummy tensors and call the functions to trigger builds.
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

# Step 1: Import aiter (triggers module_aiter_enum build)
print("\n[1] Import aiter...", flush=True)
t0 = time.time()
try:
    import aiter
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)
    sys.exit(1)

# Step 2: Import torch for dummy tensor creation
print("\n[2] Import torch...", flush=True)
t0 = time.time()
import torch
print(f"  Done in {time.time()-t0:.1f}s", flush=True)
print(f"  PyTorch: {torch.__version__}", flush=True)

# Step 3: Import fused_moe functions
print("\n[3] Import fused_moe...", flush=True)
t0 = time.time()
try:
    from aiter.fused_moe import (
        fused_moe, get_inter_dim, get_padded_M, get_2stage_cfgs,
    )
    from aiter import ActivationType, QuantType, dtypes
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)
    import traceback; traceback.print_exc()

# Step 4: Build sorting modules
# moe_sorting_opus_fwd and moe_sorting_fwd are built when first accessed
print("\n[4] Build sorting modules...", flush=True)
t0 = time.time()
try:
    has_opus = hasattr(aiter, 'moe_sorting_opus_fwd')
    has_sorting = hasattr(aiter, 'moe_sorting_fwd')
    print(f"  opus: {has_opus}, sorting: {has_sorting}, took {time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)
    import traceback; traceback.print_exc()

# Step 5: Build CK modules by calling get_2stage_cfgs
# This triggers loading of CK kernel modules
print("\n[5] Build CK 2-stage modules via config lookup...", flush=True)
t0 = time.time()
try:
    # Trigger cktile module build
    from aiter.fused_moe import cktile_moe_stage1, cktile_moe_stage2
    print(f"  cktile imported in {time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"  cktile import FAILED: {e}", flush=True)
    import traceback; traceback.print_exc()

t0 = time.time()
try:
    # Try to get configs for various shapes to trigger CK module compilation
    # These config lookups trigger the JIT compilation of the CK kernels
    shapes = [
        # (token, model_dim, inter_dim, expert, topk)
        (256, 7168, 256, 257, 9),   # E=257, small
        (256, 7168, 512, 33, 9),    # E=33, d_expert=512
        (256, 7168, 2048, 33, 9),   # E=33, d_expert=2048
    ]
    for token, model_dim, inter_dim, expert, topk in shapes:
        try:
            md = get_2stage_cfgs(
                token, model_dim, inter_dim * 2, expert, topk,
                torch.bfloat16,
                dtypes.fp4x2, dtypes.fp4x2,
                QuantType.per_1x32, True,
                ActivationType.Silu, False,
                0, 0, True,
            )
            print(f"  Config for (T={token}, E={expert}, D={inter_dim}): ksplit={md.ksplit}, block_m={md.block_m}", flush=True)
        except Exception as e:
            print(f"  Config lookup failed for shape: {e}", flush=True)
    print(f"  Config lookups done in {time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)
    import traceback; traceback.print_exc()

# Step 6: Try to import ck_moe_stage1/stage2 directly
print("\n[6] Build CK FP4 MOE modules...", flush=True)
t0 = time.time()
try:
    from aiter.fused_moe import ck_moe_stage1, ck_moe_stage2
    print(f"  Imported in {time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"  Direct import FAILED: {e}", flush=True)
    # Try alternative import path
    try:
        # The CK modules might only build when actually called with data
        # Since we don't have a GPU, we can try to trigger the build directly
        from aiter.jit.core import build_module
        print("  Attempting direct build...", flush=True)
    except Exception as e2:
        print(f"  Direct build also failed: {e2}", flush=True)

# Step 7: Build quant module
print("\n[7] Build quant module...", flush=True)
t0 = time.time()
try:
    # Try various import paths for the quant function
    try:
        from aiter import dynamic_mxfp4_quant
        print(f"  Imported from aiter in {time.time()-t0:.1f}s", flush=True)
    except ImportError:
        from aiter.ops.quant import dynamic_mxfp4_quant
        print(f"  Imported from aiter.ops.quant in {time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)
    # List available quant functions
    try:
        import aiter
        quant_stuff = [a for a in dir(aiter) if 'quant' in a.lower() or 'mxfp' in a.lower()]
        print(f"  Available: {quant_stuff}", flush=True)
    except:
        pass

# Step 8: Build activation module
print("\n[8] Build activation module...", flush=True)
t0 = time.time()
try:
    from aiter import silu_and_mul
    print(f"  Imported in {time.time()-t0:.1f}s", flush=True)
except ImportError:
    try:
        import aiter
        act_stuff = [a for a in dir(aiter) if 'silu' in a.lower() or 'activ' in a.lower()]
        print(f"  Available: {act_stuff}", flush=True)
    except:
        pass

# Final: List all .so files
elapsed = time.time() - start
print(f"\n=== Build complete in {elapsed:.1f}s ===", flush=True)

jit_dir = "/home/runner/aiter/aiter/jit"
print(f"\nBuilt modules in {jit_dir}:", flush=True)
if os.path.isdir(jit_dir):
    for f in sorted(os.listdir(jit_dir)):
        if f.endswith(".so"):
            size = os.path.getsize(os.path.join(jit_dir, f))
            print(f"  {f}  ({size:,} bytes)", flush=True)
else:
    print(f"  Directory not found!", flush=True)
