#!/bin/bash
# Cross-compile the MXFP4 GEMM kernel for gfx950
# Run inside the rocm-cross Docker container

set -e

echo "Compiling mxfp4_gemm.hip for gfx950..."
/opt/rocm/bin/hipcc \
    -O3 \
    -mcumode \
    --offload-arch=gfx950 \
    -shared -fPIC \
    -o mxfp4_gemm.so \
    mxfp4_gemm.hip \
    -I/opt/rocm/include \
    -L/opt/rocm/lib \
    -lamdhip64

echo "Extracting HSACO binary..."
/opt/rocm/bin/hipcc \
    -O3 \
    -mcumode \
    --offload-arch=gfx950 \
    -c \
    --save-temps=obj \
    -o mxfp4_gemm.o \
    mxfp4_gemm.hip \
    -I/opt/rocm/include

# Find the .co file
ls -la *.co *.hsaco 2>/dev/null || echo "No .co/.hsaco files found"
ls -la *.o *.so 2>/dev/null

echo "Done! Ship mxfp4_gemm.so or the .co file with your submission."
