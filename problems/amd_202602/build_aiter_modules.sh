#!/bin/bash
# Build AITER JIT modules for gfx950 in Docker
# These modules take 220s to JIT compile on the runner. Pre-building them
# eliminates the cold start penalty.
#
# Usage: ./build_aiter_modules.sh
# Output: aiter_modules/*.so

set -e

DOCKER_IMAGE="rocm/pytorch:latest"
AITER_DIR="/Users/samuelxu/dev/reference-kernels/problems/amd_202602/aiter"
OUTPUT_DIR="/Users/samuelxu/dev/reference-kernels/problems/amd_202602/aiter_modules"

mkdir -p "$OUTPUT_DIR"

echo "Building AITER JIT modules in Docker..."
echo "Image: $DOCKER_IMAGE"
echo "AITER: $AITER_DIR"

docker run --rm --platform linux/amd64 \
  -v "$AITER_DIR":/workspace/aiter \
  -v "$OUTPUT_DIR":/workspace/output \
  "$DOCKER_IMAGE" \
  bash -c '
cd /workspace/aiter

# Install aiter in develop mode
echo "=== Installing aiter ==="
pip install -e . 2>&1 | tail -5

# Trigger JIT builds by importing modules
echo "=== Building JIT modules ==="
python3 -c "
import os
os.environ[\"HIP_VISIBLE_DEVICES\"] = \"\"  # No GPU needed for compilation
os.environ[\"AITER_ASM_DIR\"] = \"/workspace/aiter/hsa\"

try:
    # This triggers module_mla_metadata, module_mla_asm, module_mla_reduce builds
    from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
    print(\"MLA modules imported\")
except Exception as e:
    print(f\"MLA import error: {e}\")

try:
    # This triggers module_quant build
    import aiter
    print(\"aiter imported\")
except Exception as e:
    print(f\"aiter import error: {e}\")
" 2>&1

# Copy built .so files to output
echo "=== Copying .so files ==="
for f in /workspace/aiter/aiter/jit/*.so; do
  if [ -f "$f" ]; then
    cp "$f" /workspace/output/
    echo "Copied: $(basename $f) ($(stat -c%s "$f") bytes)"
  fi
done

echo "=== Done ==="
ls -la /workspace/output/*.so 2>/dev/null || echo "No .so files built"
'
