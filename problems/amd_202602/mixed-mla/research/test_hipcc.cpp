// Simple HIP kernel test for gfx950 cross-compilation
#include <hip/hip_runtime.h>

extern "C" __global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
