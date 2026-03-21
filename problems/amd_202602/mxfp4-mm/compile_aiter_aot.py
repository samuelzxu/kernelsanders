"""Compile aiter's actual preshuffle kernel AOT for competition shapes."""
import sys, os, json, base64
sys.path.insert(0, os.environ.get("AITER_PATH", "/workspace/problems/amd_202602/aiter"))

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource

print("Importing aiter kernel...")
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import _gemm_afp4wfp4_preshuffle_kernel

target = GPUTarget("hip", "gfx950", 64)
backend = triton.compiler.make_backend(target)
print(f"Backend: {backend.binary_ext}")

kernel_fn = _gemm_afp4wfp4_preshuffle_kernel

configs = [
    ("k1536_m256", {
        "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 4, "NUM_KSPLIT": 3, "SPLITK_BLOCK_SIZE": 512,
        "EVEN_K": True, "num_warps": 8, "num_stages": 2,
        "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
    }),
    ("k2048_m64", {
        "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "SPLITK_BLOCK_SIZE": 1024,
        "EVEN_K": True, "num_warps": 8, "num_stages": 2,
        "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg",
    }),
]

for name, constexprs in configs:
    print(f"\n=== Compiling {name} ===")

    sig = {}
    for arg_name in kernel_fn.fn.arg_names:
        if arg_name in constexprs:
            sig[arg_name] = "constexpr"
        elif "ptr" in arg_name:
            sig[arg_name] = "*u8" if ("scale" in arg_name or arg_name in ("a_ptr", "b_ptr")) else "*bf16"
        elif "stride" in arg_name or arg_name in ("M", "N", "K"):
            sig[arg_name] = "i32"

    src = ASTSource(fn=kernel_fn.fn, constexprs=constexprs, signature=sig, attrs={})
    options = backend.parse_options({
        "num_warps": constexprs["num_warps"],
        "num_stages": constexprs["num_stages"],
        "waves_per_eu": constexprs["waves_per_eu"],
        "matrix_instr_nonkdim": constexprs["matrix_instr_nonkdim"],
    })

    try:
        print("Compiling...")
        ccinfo = triton.compile(src, target=target, options=options.__dict__)
        hsaco = ccinfo.asm.get("hsaco")
        if hsaco:
            print(f"SUCCESS! HSACO: {len(hsaco)} bytes")
            b64 = base64.b64encode(hsaco).decode()
            out_dir = f"aot_preshuffle_{name}"
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
