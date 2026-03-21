"""
AOT compilation script for gemm_afp4wfp4_preshuffle kernel.

Run inside Docker container with ROCm 7.1 + aiter + Triton:
  docker run -it -v $(pwd):/workspace atom-gfx950 bash
  cd /workspace
  python compile_aot_preshuffle.py

This compiles the preshuffle kernel for each competition shape and produces
.hsaco + .json metadata files that can be embedded in the submission.

The compiled binaries bypass ALL Triton JIT compilation at runtime.
"""
import os
import sys
import json
import base64
import shutil

# Add aiter to path
sys.path.insert(0, '/home/runner/aiter')  # adjust for Docker

import triton
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import (
    _gemm_afp4wfp4_preshuffle_kernel,
)
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import get_splitk
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH

# Target configs for each competition shape
# The preshuffle kernel uses K*2 in path names and K_packed internally
SHAPES = {
    # (M, N, K_logical) → config for the preshuffle kernel
    # These match our #211 best configs translated to afp4wfp4_preshuffle format
    (4, 2880, 512): {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1,
        "num_warps": 4, "num_stages": 1, "waves_per_eu": 1,
        "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
    },
    (16, 2112, 7168): {
        # Use aiter's existing tuned K=7168 preshuffle config
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 1, "NUM_KSPLIT": 14,
        "num_warps": 4, "num_stages": 1, "waves_per_eu": 1,
        "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
    },
    (32, 4096, 512): {
        "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1,
        "num_warps": 4, "num_stages": 1, "waves_per_eu": 2,
        "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
    },
    (32, 2880, 512): {
        "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1,
        "num_warps": 4, "num_stages": 1, "waves_per_eu": 2,
        "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
    },
    (64, 7168, 2048): {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2,
        "num_warps": 8, "num_stages": 2, "waves_per_eu": 4,
        "matrix_instr_nonkdim": 32, "cache_modifier": ".cg",
    },
    (256, 3072, 1536): {
        "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 4, "NUM_KSPLIT": 3,
        "num_warps": 8, "num_stages": 2, "waves_per_eu": 4,
        "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
    },
}

# Test shapes
TEST_SHAPES = {
    (8, 2112, 7168): SHAPES[(16, 2112, 7168)],  # same config as M=16
    (16, 3072, 1536): {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3,
        "num_warps": 4, "num_stages": 1, "waves_per_eu": 2,
        "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
    },
    (64, 3072, 1536): {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3,
        "num_warps": 4, "num_stages": 1, "waves_per_eu": 2,
        "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
    },
    (256, 2880, 512): {
        "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1,
        "num_warps": 4, "num_stages": 1, "waves_per_eu": 2,
        "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
    },
}


def compile_shape(m, n, k_logical, config):
    """Compile AOT binary for a specific shape."""
    import torch

    k_packed = k_logical // 2
    kernel_name = _gemm_afp4wfp4_preshuffle_kernel.fn.__name__

    # Compute metadata path (matches what gemm_afp4wfp4_preshuffle expects)
    m_pow2 = triton.next_power_of_2(m)
    if m < 32 and m_pow2 > 16:
        m_pow2 = 16
    metadata_dir = f"aot_output/{kernel_name}_M={m_pow2}-N={n}-K={k_logical * 2}"

    print(f"\n{'='*60}")
    print(f"Compiling: M={m}, N={n}, K={k_logical} (path: M={m_pow2}-N={n}-K={k_logical*2})")
    print(f"Config: {config}")

    # Apply get_splitk adjustment
    full_config = dict(config)
    if full_config["NUM_KSPLIT"] > 1:
        sb, bk, ns = get_splitk(k_packed, full_config["BLOCK_SIZE_K"], full_config["NUM_KSPLIT"])
        full_config["SPLITK_BLOCK_SIZE"] = sb
        full_config["BLOCK_SIZE_K"] = bk
        full_config["NUM_KSPLIT"] = ns
    else:
        full_config["SPLITK_BLOCK_SIZE"] = 2 * k_packed

    # Create dummy tensors to trigger compilation
    A_q = torch.empty((m, k_packed), dtype=torch.uint8, device="cuda")
    B_q = torch.empty((n, k_packed), dtype=torch.uint8, device="cuda")
    A_scale = torch.empty((m, k_logical // 32), dtype=torch.uint8, device="cuda")
    B_scale = torch.empty((n, k_logical // 32), dtype=torch.uint8, device="cuda")

    # Reshape B for preshuffle: (N, K//2) -> (N//16, K*8)
    B_w = B_q.reshape(n // 16, k_packed * 16)

    # Reshape scales for preshuffle: (N, K//32) -> (N//32, K)
    B_scale_ps = B_scale.reshape(n // 32, k_logical) if n >= 32 else B_scale

    B_w_t = B_w.T.contiguous()  # kernel transposes internally

    ksplit = full_config["NUM_KSPLIT"]
    if ksplit == 1:
        y = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    else:
        y = torch.empty((ksplit, m, n), dtype=torch.float32, device="cuda")

    grid = (
        ksplit * triton.cdiv(m, full_config["BLOCK_SIZE_M"]) * triton.cdiv(n, full_config["BLOCK_SIZE_N"]),
    )

    print(f"Grid: {grid}, SPLITK_BLOCK_SIZE: {full_config['SPLITK_BLOCK_SIZE']}")

    # Launch kernel to trigger JIT compilation
    try:
        _gemm_afp4wfp4_preshuffle_kernel[grid](
            A_q, B_w_t,
            y,
            A_scale, B_scale_ps,
            m, n, k_packed,
            A_q.stride(0), A_q.stride(1),
            B_w_t.stride(0), B_w_t.stride(1),
            0 if ksplit == 1 else y.stride(0),
            y.stride(0) if ksplit == 1 else y.stride(1),
            y.stride(1) if ksplit == 1 else y.stride(2),
            A_scale.stride(0), A_scale.stride(1),
            B_scale_ps.stride(0), B_scale_ps.stride(1),
            **full_config,
        )
        torch.cuda.synchronize()
        print("Compilation successful!")
    except Exception as e:
        print(f"Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Find the compiled binary in Triton cache
    cache_dir = os.path.expanduser("~/.triton/cache")
    hsaco_files = []
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            if f.endswith('.hsaco'):
                path = os.path.join(root, f)
                hsaco_files.append((os.path.getmtime(path), path))

    if not hsaco_files:
        print("No .hsaco files found!")
        return None

    hsaco_files.sort(reverse=True)
    latest_hsaco = hsaco_files[0][1]

    # Also find the metadata JSON
    metadata_json_path = latest_hsaco.replace('.hsaco', '.json')
    if not os.path.exists(metadata_json_path):
        # Try in the same directory
        hsaco_dir = os.path.dirname(latest_hsaco)
        json_files = [f for f in os.listdir(hsaco_dir) if f.endswith('.json')]
        if json_files:
            metadata_json_path = os.path.join(hsaco_dir, json_files[0])
        else:
            print("No metadata JSON found!")
            return None

    # Copy to output directory
    os.makedirs(metadata_dir, exist_ok=True)
    out_hsaco = os.path.join(metadata_dir, f"{kernel_name}.hsaco")
    out_json = os.path.join(metadata_dir, f"{kernel_name}.json")
    shutil.copy2(latest_hsaco, out_hsaco)
    shutil.copy2(metadata_json_path, out_json)

    print(f"Saved to {metadata_dir}/")
    print(f"  .hsaco: {os.path.getsize(out_hsaco)} bytes")
    print(f"  .json: {os.path.getsize(out_json)} bytes")

    # Base64 encode
    with open(out_hsaco, 'rb') as f:
        hsaco_b64 = base64.b64encode(f.read()).decode()
    with open(out_json, 'r') as f:
        json_content = f.read()

    return {
        "metadata_dir": metadata_dir,
        "hsaco_b64": hsaco_b64,
        "json_content": json_content,
        "hsaco_size": os.path.getsize(out_hsaco),
    }


def main():
    all_shapes = {**SHAPES, **TEST_SHAPES}
    results = {}

    for (m, n, k), config in all_shapes.items():
        result = compile_shape(m, n, k, config)
        if result:
            results[(m, n, k)] = result

    # Print summary
    print(f"\n{'='*60}")
    print("COMPILATION SUMMARY")
    print(f"{'='*60}")
    total_size = 0
    for (m, n, k), r in results.items():
        print(f"M={m:3d} N={n:4d} K={k:4d}: {r['hsaco_size']:6d} bytes, b64={len(r['hsaco_b64']):6d} chars")
        total_size += len(r['hsaco_b64'])

    print(f"\nTotal base64 size: {total_size:,d} chars")
    print(f"Estimated submission overhead: ~{total_size // 1024}KB")

    # Save all base64 for embedding
    with open("aot_output/all_kernels.json", "w") as f:
        out = {}
        for (m, n, k), r in results.items():
            key = f"M={m}-N={n}-K={k}"
            out[key] = {
                "hsaco_b64": r["hsaco_b64"],
                "json": json.loads(r["json_content"]),
            }
        json.dump(out, f, indent=2)
    print(f"\nAll kernels saved to aot_output/all_kernels.json")


if __name__ == "__main__":
    main()
