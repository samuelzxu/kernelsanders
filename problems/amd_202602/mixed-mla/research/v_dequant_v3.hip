// MXFP4 V dequantization v3 — with stride support
// V data is at the FIRST DV_HALF bytes of each token's DQ_HALF-byte row
// Stride between tokens = DQ_HALF (288), not DV_HALF (256)

#include <hip/hip_runtime.h>

__device__ __constant__ float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__global__ void mxfp4_v_to_bf16_kernel(
    const unsigned char* __restrict__ V_fp4,
    const unsigned char* __restrict__ V_scale,
    unsigned short* __restrict__ V_out,
    int n, int DV_HALF, int DV, int SCALE_GROUP,
    int stride_fp4, int stride_scale  // byte strides between tokens
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * DV_HALF;
    if (idx >= total) return;

    int token = idx / DV_HALF;
    int byte_pos = idx % DV_HALF;

    // Use stride to index into the correct token's data
    unsigned char packed = V_fp4[token * stride_fp4 + byte_pos];

    int scale_idx = byte_pos / (SCALE_GROUP / 2);
    unsigned char scale_e8m0 = V_scale[token * stride_scale + scale_idx];

    // E8M0 to float: value = 2^(scale - 127)
    // In IEEE 754: exponent field = scale, so float bits = scale << 23
    int exp_bits = ((int)scale_e8m0) << 23;
    float scale = __int_as_float(exp_bits);

    float val_lo = FP4_LUT[packed & 0xFu] * scale;
    float val_hi = FP4_LUT[packed >> 4] * scale;

    unsigned int lo_bits = __float_as_int(val_lo);
    unsigned int hi_bits = __float_as_int(val_hi);
    V_out[token * DV + 2 * byte_pos] = (unsigned short)(lo_bits >> 16);
    V_out[token * DV + 2 * byte_pos + 1] = (unsigned short)(hi_bits >> 16);
}

extern "C" int launch_v_dequant(
    void* v_fp4_ptr,
    void* v_scale_ptr,
    void* v_out_ptr,
    int n,
    int DV_HALF,
    int DV,
    int SCALE_GROUP,
    int stride_fp4,    // token stride in bytes for fp4 data
    int stride_scale,  // token stride in bytes for scale data
    void* hip_strm_ptr
) {
    int total = n * DV_HALF;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    hipStream_t hstrm = (hipStream_t)hip_strm_ptr;

    mxfp4_v_to_bf16_kernel<<<dim3(grid_size), dim3(block_size), 0, hstrm>>>(
        (const unsigned char*)v_fp4_ptr,
        (const unsigned char*)v_scale_ptr,
        (unsigned short*)v_out_ptr,
        n, DV_HALF, DV, SCALE_GROUP,
        stride_fp4, stride_scale
    );

    return (int)hipGetLastError();
}
