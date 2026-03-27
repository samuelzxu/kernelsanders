"""Standalone gluon kernel compiler - mocks all aiter deps, works with Triton 3.6+rocm7.2."""
import sys, os, types, importlib.util, json, base64

# Mock aiter module hierarchy
for mod_name in [
    "aiter", "aiter.ops", "aiter.ops.triton", "aiter.ops.triton.utils",
    "aiter.ops.triton.utils._triton", "aiter.ops.triton.utils._triton.arch_info",
    "aiter.ops.triton.utils._triton.pid_preprocessing",
    "aiter.ops.triton.utils._triton.kernel_repr",
    "aiter.ops.triton.utils.core", "aiter.ops.triton.utils.logger",
    "aiter.ops.triton.utils.gemm_config_utils",
]:
    sys.modules[mod_name] = types.ModuleType(mod_name)

# Mock arch_info
ai = sys.modules["aiter.ops.triton.utils._triton.arch_info"]
ai.is_fp4_avail = lambda: True
ai.get_arch = lambda: "gfx950"
sys.modules["aiter.ops.triton.utils._triton"].arch_info = ai

# Mock core
sys.modules["aiter.ops.triton.utils.core"].AITER_TRITON_CONFIGS_PATH = "/tmp/configs"

# Mock logger
class FakeLogger:
    def info(self, *a, **kw): pass
    def debug(self, *a, **kw): pass
    def warning(self, *a, **kw): pass
sys.modules["aiter.ops.triton.utils.logger"].AiterTritonLogger = FakeLogger

# Load pid_preprocessing from extracted aiter source
pp_path = "/aiter/aiter/ops/triton/utils/pid_preprocessing.py"
if os.path.exists(pp_path):
    spec = importlib.util.spec_from_file_location("aiter.ops.triton.utils._triton.pid_preprocessing", pp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aiter.ops.triton.utils._triton.pid_preprocessing"] = mod
    spec.loader.exec_module(mod)
    print(f"Loaded pid_preprocessing from {pp_path}")
else:
    # Inline minimal pid_grid and remap_xcd
    import triton
    import triton.language as tl
    pp = sys.modules["aiter.ops.triton.utils._triton.pid_preprocessing"]

    @triton.jit
    def pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr):
        if GROUP_SIZE_M > 0:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
            pid_n = (pid % num_pid_in_group) // group_size_m
        else:
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n
        return pid_m, pid_n

    @triton.jit
    def remap_xcd(pid, NUM_XCDS: tl.constexpr):
        return pid

    pp.pid_grid = pid_grid
    pp.remap_xcd = remap_xcd
    print("Using inline pid_grid/remap_xcd")

# Mock kernel_repr
kr = sys.modules["aiter.ops.triton.utils._triton.kernel_repr"]
kr.kernel_repr_config = lambda **kw: lambda fn: fn

# Mock gemm_config_utils
gc = sys.modules["aiter.ops.triton.utils.gemm_config_utils"]
gc.get_gemm_config = lambda *a, **kw: ({}, None)

# Now load the gluon kernel
print("Loading gluon kernel...")
gluon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gluon_kernel_patched.py")
if not os.path.exists(gluon_path):
    gluon_path = "/workspace/gluon_kernel_patched.py"
spec = importlib.util.spec_from_file_location("gluon_kernel", gluon_path)
gmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gmod)

kernel = gmod._gemm_afp4wfp4_kernel
print(f"Kernel: {kernel}")
print(f"Args: {kernel.fn.arg_names}")

# Compile
import triton
from triton.compiler.compiler import GPUTarget

target = GPUTarget("hip", "gfx950", 64)
backend = triton.compiler.make_backend(target)
print(f"Backend ext: {backend.binary_ext}")

try:
    from triton.experimental.gluon._compiler import GluonASTSource
    print("Using GluonASTSource from gluon._compiler")
except ImportError:
    try:
        from triton.experimental.gluon._runtime import GluonASTSource
        print("Using GluonASTSource from gluon._runtime")
    except ImportError:
        from triton.compiler.compiler import ASTSource as GluonASTSource
        print("WARNING: Using generic ASTSource")

configs = [
    ("k1536_m256", {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1536,
        "EVEN_K": True, "num_warps": 8, "num_stages": 2,
        "waves_per_eu": 0, "matrix_instr_nonkdim": 32, "cache_modifier": None,
    }),
]

for name, constexprs in configs:
    print(f"\n=== Compiling {name} ===")
    sig = {}
    for arg_name in kernel.fn.arg_names:
        if arg_name in constexprs:
            sig[arg_name] = "constexpr"
        elif "ptr" in arg_name:
            sig[arg_name] = "*u8"
        elif "stride" in arg_name or arg_name in ("M", "N", "K"):
            sig[arg_name] = "i32"

    print(f"Signature: {sig}")
    print(f"Constexprs: {constexprs}")

    try:
        src = GluonASTSource(fn=kernel.fn, constexprs=constexprs, signature=sig, attrs={})
        compiled = triton.compile(src, target=target)
        print(f"Compiled! Binary size: {len(compiled.asm.get('hsaco', b''))} bytes")

        # Save
        out_dir = f"/workspace/aot_gluon_{name}"
        os.makedirs(out_dir, exist_ok=True)
        hsaco = compiled.asm.get("hsaco", b"")
        with open(f"{out_dir}/kernel.hsaco", "wb") as f:
            f.write(hsaco)
        with open(f"{out_dir}/kernel_b64.txt", "w") as f:
            f.write(base64.b64encode(hsaco).decode())
        meta = {"name": compiled.name, "num_warps": constexprs["num_warps"],
                "shared": compiled.shared, "args": list(sig.keys())}
        with open(f"{out_dir}/meta.json", "w") as f:
            json.dump(meta, f)
        print(f"Saved to {out_dir}")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()
