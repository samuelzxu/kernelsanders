"""AOT compile gluon FP4 GEMM kernel — bypass full aiter import."""
import sys, os, json, base64, importlib.util

# Manually import just the gluon kernel file without triggering full aiter init
aiter_root = os.environ.get("AITER_PATH", "/workspace/problems/amd_202602/aiter")
sys.path.insert(0, aiter_root)

# We need these sub-modules loaded manually:
# 1. triton (already available)
# 2. aiter.ops.triton.utils._triton.pid_preprocessing (for remap_xcd, pid_grid)
# 3. aiter.ops.triton.gluon.gemm_afp4wfp4 (the actual kernel)

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget

# Manually set up minimal aiter module structure
import types
aiter_mod = types.ModuleType("aiter")
aiter_ops = types.ModuleType("aiter.ops")
aiter_triton = types.ModuleType("aiter.ops.triton")
aiter_utils = types.ModuleType("aiter.ops.triton.utils")
aiter_triton_inner = types.ModuleType("aiter.ops.triton.utils._triton")
sys.modules["aiter"] = aiter_mod
sys.modules["aiter.ops"] = aiter_ops
sys.modules["aiter.ops.triton"] = aiter_triton
sys.modules["aiter.ops.triton.utils"] = aiter_utils
# Pre-load arch_info into the _triton namespace
ai_mod_early = types.ModuleType("aiter.ops.triton.utils._triton.arch_info")
ai_mod_early.is_fp4_avail = lambda: True
ai_mod_early.get_arch = lambda: "gfx950"
aiter_triton_inner.arch_info = ai_mod_early
sys.modules["aiter.ops.triton.utils._triton"] = aiter_triton_inner
sys.modules["aiter.ops.triton.utils._triton.arch_info"] = ai_mod_early

# Load pid_preprocessing
pp_path = f"{aiter_root}/aiter/ops/triton/utils/_triton/pid_preprocessing.py"
pp_spec = importlib.util.spec_from_file_location("aiter.ops.triton.utils._triton.pid_preprocessing", pp_path)
pp_mod = importlib.util.module_from_spec(pp_spec)
sys.modules["aiter.ops.triton.utils._triton.pid_preprocessing"] = pp_mod
pp_spec.loader.exec_module(pp_mod)

# Load kernel_repr (needed by gluon kernel)
kr_path = f"{aiter_root}/aiter/ops/triton/utils/_triton/kernel_repr.py"
kr_spec = importlib.util.spec_from_file_location("aiter.ops.triton.utils._triton.kernel_repr", kr_path)
kr_mod = importlib.util.module_from_spec(kr_spec)
sys.modules["aiter.ops.triton.utils._triton.kernel_repr"] = kr_mod
kr_spec.loader.exec_module(kr_mod)

# Load gemm_config_utils
gc_path = f"{aiter_root}/aiter/ops/triton/utils/gemm_config_utils.py"
gc_spec = importlib.util.spec_from_file_location("aiter.ops.triton.utils.gemm_config_utils", gc_path)
gc_mod = importlib.util.module_from_spec(gc_spec)
sys.modules["aiter.ops.triton.utils.gemm_config_utils"] = gc_mod
# Mock AITER_TRITON_CONFIGS_PATH
core_mod = types.ModuleType("aiter.ops.triton.utils.core")
core_mod.AITER_TRITON_CONFIGS_PATH = f"{aiter_root}/aiter/ops/triton/configs"
sys.modules["aiter.ops.triton.utils.core"] = core_mod
gc_spec.loader.exec_module(gc_mod)

# Load arch_info
ai_path = f"{aiter_root}/aiter/ops/triton/utils/_triton/arch_info.py"
ai_spec = importlib.util.spec_from_file_location("aiter.ops.triton.utils._triton.arch_info", ai_path)
ai_mod = importlib.util.module_from_spec(ai_spec)
sys.modules["aiter.ops.triton.utils._triton.arch_info"] = ai_mod
# Mock the arch functions
ai_mod.is_fp4_avail = lambda: True
ai_mod.get_arch = lambda: "gfx950"
sys.modules["aiter.ops.triton.utils._triton"] = aiter_triton_inner
aiter_triton_inner.arch_info = ai_mod

# Load logger
logger_mod = types.ModuleType("aiter.ops.triton.utils.logger")
class FakeLogger:
    def info(self, *a, **kw): pass
    def debug(self, *a, **kw): pass
    def warning(self, *a, **kw): pass
logger_mod.AiterTritonLogger = FakeLogger
sys.modules["aiter.ops.triton.utils.logger"] = logger_mod

# Now load the PATCHED gluon kernel (instr_shape=[32,32,64] for Triton 3.6)
print("Loading patched gluon kernel...")
gluon_path = os.path.join(os.path.dirname(__file__), "gluon_kernel_patched.py")
gluon_spec = importlib.util.spec_from_file_location("aiter.ops.triton.gluon.gemm_afp4wfp4", gluon_path)
gluon_mod = importlib.util.module_from_spec(gluon_spec)
sys.modules["aiter.ops.triton.gluon.gemm_afp4wfp4"] = gluon_mod
gluon_spec.loader.exec_module(gluon_mod)

gluon_kernel = gluon_mod._gemm_afp4wfp4_kernel
print(f"Kernel loaded: {gluon_kernel}")
print(f"Arg names: {gluon_kernel.fn.arg_names}")

target = GPUTarget("hip", "gfx950", 64)
backend = triton.compiler.make_backend(target)
print(f"Backend: {backend.binary_ext}")

# Gluon needs GluonASTSource
try:
    from triton.experimental.gluon._runtime import GluonASTSource
    print("Using GluonASTSource")
except ImportError:
    from triton.compiler.compiler import ASTSource as GluonASTSource
    print("WARNING: GluonASTSource not available, using ASTSource")

configs = [
    ("k1536_m256", {
        "BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1536,
        "EVEN_K": True, "num_warps": 8, "num_stages": 2,
        "waves_per_eu": 0, "matrix_instr_nonkdim": 32, "cache_modifier": None,
    }),
    ("k2048_m64", {
        "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 2048,
        "EVEN_K": True, "num_warps": 8, "num_stages": 2,
        "waves_per_eu": 0, "matrix_instr_nonkdim": 32, "cache_modifier": None,
    }),
]

for name, constexprs in configs:
    print(f"\n=== Compiling gluon {name} ===")

    sig = {}
    for arg_name in gluon_kernel.fn.arg_names:
        if arg_name in constexprs:
            sig[arg_name] = "constexpr"
        elif "ptr" in arg_name:
            sig[arg_name] = "*u8" if ("scale" in arg_name or arg_name in ("a_ptr", "b_ptr")) else "*bf16"
        elif "stride" in arg_name or arg_name in ("M", "N", "K"):
            sig[arg_name] = "i32"

    src = GluonASTSource(fn=gluon_kernel.fn, constexprs=constexprs, signature=sig, attrs={})
    options = backend.parse_options({
        "num_warps": constexprs["num_warps"],
        "num_stages": constexprs["num_stages"],
        "waves_per_eu": constexprs["waves_per_eu"],
        "matrix_instr_nonkdim": constexprs["matrix_instr_nonkdim"],
    })

    try:
        print("Compiling (gluon is slower than standard Triton)...")
        ccinfo = triton.compile(src, target=target, options=options.__dict__)
        hsaco = ccinfo.asm.get("hsaco")
        if hsaco:
            print(f"SUCCESS! HSACO: {len(hsaco)} bytes")
            b64 = base64.b64encode(hsaco).decode()
            out_dir = f"aot_gluon_{name}"
            os.makedirs(out_dir, exist_ok=True)
            with open(f"{out_dir}/kernel.hsaco", "wb") as f:
                f.write(hsaco)
            with open(f"{out_dir}/kernel.b64", "w") as f:
                f.write(b64)
            metadata = {}
            for k in dir(ccinfo.metadata):
                if not k.startswith("_"):
                    try:
                        v = getattr(ccinfo.metadata, k)
                        if callable(v): continue
                        json.dumps(v)
                        metadata[k] = v
                    except (TypeError, ValueError):
                        metadata[k] = str(getattr(ccinfo.metadata, k))
            with open(f"{out_dir}/kernel.json", "w") as f:
                json.dump(metadata, f, indent=2)
            kname = metadata.get("name", "?")
            kshared = metadata.get("shared", "?")
            print(f"Kernel: {kname}")
            print(f"Shared: {kshared}")
            print(f"B64: {len(b64)} chars")
        else:
            print(f"No hsaco! Keys: {list(ccinfo.asm.keys())}")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

print("\nDone!")
