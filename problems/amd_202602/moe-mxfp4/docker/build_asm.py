import os, sys, time
os.environ["GPU_ARCHS"] = "gfx950"
os.environ["CU_NUM"] = "256"
sys.path.insert(0, "/home/runner/aiter/aiter/jit")
sys.path.insert(0, "/home/runner/aiter/aiter/jit/utils")
import aiter
from core import build_module, get_args_of_build
t0 = time.time()
d = get_args_of_build("module_moe_asm")
build_module("module_moe_asm", d["srcs"], d["flags_extra_cc"], d["flags_extra_hip"],
    d["blob_gen_cmd"], d["extra_include"], d["extra_ldflags"], d["verbose"],
    d["is_python_module"], d["is_standalone"], d["torch_exclude"],
    d.get("third_party", []), hipify=d.get("hipify", False))
print(f"Done in {time.time()-t0:.1f}s")
for f in sorted(os.listdir("/home/runner/aiter/aiter/jit")):
    if f.endswith(".so"):
        sz = os.path.getsize(f"/home/runner/aiter/aiter/jit/{f}")
        print(f"  {f} ({sz:,} bytes)")
