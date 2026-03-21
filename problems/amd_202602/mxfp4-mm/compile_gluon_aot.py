"""AOT compile aiter's gluon FP4 GEMM kernel for gfx950. Run in Docker."""
import sys, os, json, base64
sys.path.insert(0, os.environ.get("AITER_PATH", "/workspace/problems/amd_202602/aiter"))

import triton
from triton.backends.compiler import GPUTarget
try:
    from triton.experimental.gluon._runtime import GluonASTSource
except ImportError:
    from triton.compiler.compiler import ASTSource as GluonASTSource
    print("WARNING: Using ASTSource instead of GluonASTSource", file=sys.stderr)

print("Importing gluon kernel...")
from aiter.ops.triton.gluon.gemm_afp4wfp4 import _gemm_afp4wfp4_kernel as gluon_kernel

target = GPUTarget("hip", "gfx950", 64)
backend = triton.compiler.make_backend(target)
print(f"Backend: {backend.binary_ext}")
print(f"Kernel arg names: {gluon_kernel.fn.arg_names}")

# The gluon kernel has these constexprs:
# BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
# NUM_KSPLIT, SPLITK_BLOCK_SIZE, EVEN_K,
# num_warps, num_stages, waves_per_eu, matrix_instr_nonkdim, cache_modifier

# Configs for our competition shapes
# Note: gluon kernel requires BSM >= 64 (warps_per_cta=[2, nw//2])
# and always uses mfma 32x32 internally
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
    print(f"Config: BSM={constexprs['BLOCK_SIZE_M']} BSN={constexprs['BLOCK_SIZE_N']} BSK={constexprs['BLOCK_SIZE_K']} KSPLIT={constexprs['NUM_KSPLIT']}")

    sig = {}
    for arg_name in gluon_kernel.fn.arg_names:
        if arg_name in constexprs:
            sig[arg_name] = "constexpr"
        elif "ptr" in arg_name:
            sig[arg_name] = "*u8" if ("scale" in arg_name or arg_name in ("a_ptr", "b_ptr")) else "*bf16"
        elif "stride" in arg_name or arg_name in ("M", "N", "K"):
            sig[arg_name] = "i32"

    try:
        src = GluonASTSource(fn=gluon_kernel.fn, constexprs=constexprs, signature=sig, attrs={})
    except Exception as e:
        print(f"GluonASTSource failed: {e}")
        print("Trying standard ASTSource...")
        from triton.compiler.compiler import ASTSource
        src = ASTSource(fn=gluon_kernel.fn, constexprs=constexprs, signature=sig, attrs={})

    options = backend.parse_options({
        "num_warps": constexprs["num_warps"],
        "num_stages": constexprs["num_stages"],
        "waves_per_eu": constexprs["waves_per_eu"],
        "matrix_instr_nonkdim": constexprs["matrix_instr_nonkdim"],
    })

    try:
        print("Compiling (gluon kernels take longer)...")
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
                        if callable(v):
                            continue
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
