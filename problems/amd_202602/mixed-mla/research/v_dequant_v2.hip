// MXFP4 V dequantization with host-callable wrapper
// Compile: hipcc -O3 --offload-arch=gfx950 -shared -fPIC v_dequant_v2.hip -o v_dequant_v2.so
// Call from Python: lib = ctypes.CDLL("v_dequant_v2.so"); lib.launch_v_dequant(...)

#include <hip/hip_runtime.h>

// FP4 E2M1 lookup table
__device__ __constant__ float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__global__ void mxfp4_v_to_bf16_kernel(
    const unsigned char* __restrict__ V_fp4,
    const unsigned char* __restrict__ V_scale,
    unsigned short* __restrict__ V_out,  // bf16 as uint16
    int n, int DV_HALF, int DV, int SCALE_GROUP
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * DV_HALF;
    if (idx >= total) return;

    int token = idx / DV_HALF;
    int byte_pos = idx % DV_HALF;

    unsigned char packed = V_fp4[idx];
    int scale_idx = byte_pos / (SCALE_GROUP / 2);
    int n_scales = DV / SCALE_GROUP;
    unsigned char scale_e8m0 = V_scale[token * n_scales + scale_idx];

    int exp_bits = ((int)scale_e8m0) << 23;
    float scale = __int_as_float(exp_bits);

    // Dequant and convert to bf16 (stored as uint16)
    float val_lo = FP4_LUT[packed & 0xFu] * scale;
    float val_hi = FP4_LUT[packed >> 4] * scale;

    // Float32 to bf16: truncate lower 16 bits
    unsigned int lo_bits = __float_as_int(val_lo);
    unsigned int hi_bits = __float_as_int(val_hi);
    V_out[token * DV + 2 * byte_pos] = (unsigned short)(lo_bits >> 16);
    V_out[token * DV + 2 * byte_pos + 1] = (unsigned short)(hi_bits >> 16);
}

// Host-callable wrapper — call this from Python via ctypes
extern "C" int launch_v_dequant(
    void* v_fp4_ptr,     // device ptr: (n, DV_HALF) uint8
    void* v_scale_ptr,   // device ptr: (n, N_V_SCALES) uint8
    void* v_out_ptr,     // device ptr: (n, DV) bf16 (as uint16)
    int n,               // number of tokens
    int DV_HALF,         // 256
    int DV,              // 512
    int SCALE_GROUP,     // 32
    void* hip_stream_ptr // hipStream_t (pass 0 for default)
) {
    int total = n * DV_HALF;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    hipStream_t hstream = (hipStream_t)hip_stream_ptr;

    mxfp4_v_to_bf16_kernel<<<dim3(grid_size), dim3(block_size), 0, hstream>>>(
        (const unsigned char*)v_fp4_ptr,
        (const unsigned char*)v_scale_ptr,
        (unsigned short*)v_out_ptr,
        n, DV_HALF, DV, SCALE_GROUP
    );

    return (int)hipGetLastError();
}
