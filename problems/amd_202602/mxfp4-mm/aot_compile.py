"""AOT compile FP4 GEMM kernel for gfx950. Run in Docker."""
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource
import json, base64, os, sys

# ---- Kernel helpers (inlined from aiter to avoid aiter import) ----

@triton.jit
def _remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid
    return pid

@triton.jit
def _pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr = 1):
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tl.assume(group_size_m >= 0)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n

# ---- FP4 GEMM kernel with split-K, remap_xcd, CDNA4 scale unswizzle ----

@triton.heuristics({"EVEN_K": lambda args: (args["K"] % (args["BLOCK_SIZE_K"] // 2) == 0)})
@triton.jit
def fp4_gemm_cdna4(
    a_ptr, b_ptr, c_ptr, a_scales_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_ck, stride_cm, stride_cn,
    stride_asm, stride_ask, stride_bsn, stride_bsk,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr, SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr, matrix_instr_nonkdim: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    SCALE_GROUP_SIZE: tl.constexpr = 32
    GRID_MN = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    pid_unified = tl.program_id(axis=0)
    pid_unified = _remap_xcd(pid_unified, GRID_MN * NUM_KSPLIT, NUM_XCDS=8)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    if NUM_KSPLIT == 1:
        pid_m, pid_n = _pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    if (pid_k * SPLITK_BLOCK_SIZE // 2) < K:
        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE // 2, BLOCK_SIZE_K // 2)
        offs_k = tl.arange(0, BLOCK_SIZE_K // 2)
        offs_k_split = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_k
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        offs_asn = (pid_n * (BLOCK_SIZE_N // 32) + tl.arange(0, (BLOCK_SIZE_N // 32))) % N
        offs_ks = (pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE) * 32) + tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE * 32)
        b_scale_ptrs = b_scales_ptr + offs_asn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk
        offs_asm = (pid_m * (BLOCK_SIZE_M // 32) + tl.arange(0, (BLOCK_SIZE_M // 32))) % M
        a_scale_ptrs = a_scales_ptr + offs_asm[:, None] * stride_asm + offs_ks[None, :] * stride_ask

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
            a_scales = tl.load(a_scale_ptrs).reshape(
                BLOCK_SIZE_M // 32, BLOCK_SIZE_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2, 1
            ).permute(0, 5, 3, 1, 4, 2, 6).reshape(BLOCK_SIZE_M, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
            b_scales = tl.load(b_scale_ptrs, cache_modifier=cache_modifier).reshape(
                BLOCK_SIZE_N // 32, BLOCK_SIZE_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2, 1
            ).permute(0, 5, 3, 1, 4, 2, 6).reshape(BLOCK_SIZE_N, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            accumulator += tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1")
            a_ptrs += (BLOCK_SIZE_K // 2) * stride_ak
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
            a_scale_ptrs += BLOCK_SIZE_K * stride_ask
            b_scale_ptrs += BLOCK_SIZE_K * stride_bsk

        c = accumulator.to(c_ptr.type.element_ty)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :] + pid_k * stride_ck
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask, cache_modifier=".wt")


# ---- Compilation entry point ----

if __name__ == "__main__":
    target = GPUTarget('hip', 'gfx950', 64)
    backend = triton.compiler.make_backend(target)
    print(f'Backend: {backend.binary_ext}')

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
        for arg_name in fp4_gemm_cdna4.fn.arg_names:
            if arg_name in constexprs:
                sig[arg_name] = "constexpr"
            elif "ptr" in arg_name:
                sig[arg_name] = "*u8" if ("scale" in arg_name or arg_name in ("a_ptr", "b_ptr")) else "*bf16"
            elif "stride" in arg_name or arg_name in ("M", "N", "K"):
                sig[arg_name] = "i32"

        src = ASTSource(fn=fp4_gemm_cdna4.fn, constexprs=constexprs, signature=sig, attrs={})
        options = backend.parse_options({
            "num_warps": constexprs["num_warps"],
            "num_stages": constexprs["num_stages"],
            "waves_per_eu": constexprs["waves_per_eu"],
            "matrix_instr_nonkdim": constexprs["matrix_instr_nonkdim"],
        })

        try:
            print("Compiling (this may take several minutes under emulation)...")
            ccinfo = triton.compile(src, target=target, options=options.__dict__)
            hsaco = ccinfo.asm.get("hsaco")
            if hsaco:
                print(f"SUCCESS! HSACO: {len(hsaco)} bytes")
                b64 = base64.b64encode(hsaco).decode()
                out_dir = f"aot_{name}"
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
                print(f"Kernel: {ccinfo.metadata.name}")
                print(f"Shared: {ccinfo.metadata.shared}")
                print(f"B64: {len(b64)} chars")
            else:
                print(f"No hsaco! Keys: {list(ccinfo.asm.keys())}")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone!")
