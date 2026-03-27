"""
Build the CK FP4 2-stage MOE module using proper gen_func path.
Must be run in Docker with ROCm 7.1 and AITER installed.
"""
import os
import sys
import time

os.environ["GPU_ARCHS"] = "gfx950"
os.environ["CU_NUM"] = "256"

AITER_DIR = "/home/runner/aiter"
sys.path.insert(0, f"{AITER_DIR}/aiter/jit")
sys.path.insert(0, f"{AITER_DIR}/aiter/jit/utils")

import aiter
from core import build_module, get_args_of_build

# Get base build args for module_moe_ck2stages
base_args = get_args_of_build("module_moe_ck2stages")
print(f"Base srcs: {base_args['srcs']}", flush=True)
print(f"Base blob_gen_cmd: {base_args['blob_gen_cmd']}", flush=True)

# The actual module name and blob_gen_cmd
AITER_CSRC_DIR = f"{AITER_DIR}/csrc"
md_name = "module_moe_ck2stages_fp4x2_fp4x2_preshuffle_on_b16_silu_per_1x32_mulWeightStage2_"
blob_gen_cmd = [
    f"{AITER_CSRC_DIR}/ck_gemm_moe_2stages_codegen/gen_instances.py -a fp4x2 -b fp4x2 -c b16 -q per_1x32 -act silu -m 2 --preshuffle -w {{}}"
]

print(f"\n=== Building {md_name} ===", flush=True)
print(f"blob_gen_cmd: {blob_gen_cmd}", flush=True)
t0 = time.time()

try:
    build_module(
        md_name,
        base_args["srcs"],
        base_args["flags_extra_cc"],
        base_args["flags_extra_hip"],
        blob_gen_cmd,
        base_args["extra_include"],
        base_args["extra_ldflags"],
        base_args["verbose"],
        base_args["is_python_module"],
        base_args["is_standalone"],
        base_args["torch_exclude"],
        base_args.get("third_party", []),
        hipify=base_args.get("hipify", False),
    )
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)
    import traceback; traceback.print_exc()

# List results
jit_dir = "/home/runner/aiter/aiter/jit"
print(f"\nModules in {jit_dir}:", flush=True)
for f in sorted(os.listdir(jit_dir)):
    if f.endswith(".so"):
        size = os.path.getsize(os.path.join(jit_dir, f))
        print(f"  {f}  ({size:,} bytes)", flush=True)
