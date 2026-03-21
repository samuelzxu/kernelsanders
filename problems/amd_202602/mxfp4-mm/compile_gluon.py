"""
Gluon GEMM compilation script for AMD Developer Cloud.

Run on a gfx950 machine:
  python compile_gluon.py

This will:
1. Compile the gluon FP4 GEMM kernel for each target shape config
2. Save the compiled .hsaco binaries
3. Print base64-encoded binaries for embedding in submission.py

The gluon kernel uses:
- Explicit SwizzledSharedLayout for 64-bank LDS (MI355X)
- buffer_load for direct global-to-LDS transfer (128-bit/lane on CDNA4)
- mfma_scaled for hardware FP4 scaling
- remap_xcd for 8-XCD locality
"""
import sys
import os
import json
import base64
import torch
import triton

from aiter.ops.triton.gluon.gemm_afp4wfp4 import (
    _gemm_afp4wfp4_kernel,
    _gemm_afp4wfp4_reduce_kernel,
    gemm_afp4wfp4 as gluon_gemm,
)
from aiter.ops.triton.quant import dynamic_mxfp4_quant

# Target shapes
SHAPES = [
    (256, 3072, 1536),  # K=1536, M=256 — our worst shape
    (64, 7168, 2048),   # K=2048, M=64 — moderate gap
]


def compile_and_benchmark():
    """Compile gluon kernel by running it, then extract .hsaco binaries."""

    for m, n, k in SHAPES:
        print(f"\n{'='*60}")
        print(f"Shape: M={m}, N={n}, K={k}")
        print(f"{'='*60}")

        # Create test data
        A = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        B = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")

        # Quantize
        A_q, A_scale = dynamic_mxfp4_quant(A)
        B_q, B_scale = dynamic_mxfp4_quant(B)

        A_q_u8 = A_q.view(torch.uint8)
        B_q_u8 = B_q.view(torch.uint8)

        print(f"A_q: {A_q_u8.shape}, B_q: {B_q_u8.shape}")
        print(f"A_scale: {A_scale.shape}, B_scale: {B_scale.shape}")

        # Run gluon GEMM (triggers compilation)
        print("Compiling gluon kernel...")
        try:
            result = gluon_gemm(A_q_u8, B_q_u8, A_scale, B_scale, dtype=torch.bfloat16)
            torch.cuda.synchronize()
            print(f"Compilation + execution successful! Output: {result.shape}")
        except Exception as e:
            print(f"Failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Benchmark
        print("Benchmarking...")
        torch.cuda.synchronize()
        import time

        # Warmup
        for _ in range(5):
            result = gluon_gemm(A_q_u8, B_q_u8, A_scale, B_scale, dtype=torch.bfloat16)
        torch.cuda.synchronize()

        # Time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        times = []
        for _ in range(50):
            torch.cuda.synchronize()
            start.record()
            result = gluon_gemm(A_q_u8, B_q_u8, A_scale, B_scale, dtype=torch.bfloat16)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000)  # µs

        avg = sum(times) / len(times)
        mn = min(times)
        mx = max(times)
        print(f"Gluon GEMM only: {avg:.1f} ± {(mx-mn)/2:.1f} µs (min={mn:.1f}, max={mx:.1f})")

        # Full pipeline: quant + gluon
        times2 = []
        for _ in range(50):
            torch.cuda.synchronize()
            start.record()
            A_q2, A_s2 = dynamic_mxfp4_quant(A)
            result = gluon_gemm(A_q2.view(torch.uint8), B_q_u8, A_s2, B_scale, dtype=torch.bfloat16)
            end.record()
            torch.cuda.synchronize()
            times2.append(start.elapsed_time(end) * 1000)

        avg2 = sum(times2) / len(times2)
        mn2 = min(times2)
        print(f"Quant + Gluon GEMM: {avg2:.1f} µs (min={mn2:.1f})")

    # Find compiled binaries
    print(f"\n{'='*60}")
    print("Looking for compiled .hsaco files...")
    cache_dir = os.path.expanduser("~/.triton/cache")

    hsaco_files = []
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            if f.endswith('.hsaco') or f.endswith('.amdgcn'):
                path = os.path.join(root, f)
                hsaco_files.append((os.path.getmtime(path), path, os.path.getsize(path)))

    hsaco_files.sort(reverse=True)
    for mtime, path, size in hsaco_files[:10]:
        print(f"  {path} ({size} bytes)")

        # Base64 encode
        with open(path, 'rb') as fh:
            binary = fh.read()
        b64 = base64.b64encode(binary).decode()
        b64_path = path + '.b64'
        with open(b64_path, 'w') as fh:
            fh.write(b64)
        print(f"    Base64 saved to {b64_path} ({len(b64)} chars)")


if __name__ == "__main__":
    compile_and_benchmark()
