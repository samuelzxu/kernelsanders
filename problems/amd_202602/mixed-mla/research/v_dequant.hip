// Fast MXFP4 V dequantization kernel for gfx950
// Converts packed FP4 (uint8, 2 values per byte) + E8M0 scales → bf16
// Used as preprocessing before Triton attention kernel

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

// FP4 E2M1 lookup table
__device__ constexpr float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Each thread processes one packed byte (2 FP4 values)
extern "C" __global__ void mxfp4_v_to_bf16(
    const unsigned char* __restrict__ V_fp4,    // (n, DV_HALF) packed
    const unsigned char* __restrict__ V_scale,   // (n, N_V_SCALES) E8M0
    hip_bfloat16* __restrict__ V_out,            // (n, DV) bf16
    int n,          // number of tokens
    int DV_HALF,    // DV/2 = 256
    int DV,         // 512
    int SCALE_GROUP // 32 (FP4 values per scale = 16 bytes per scale)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * DV_HALF;
    if (idx >= total) return;

    int token = idx / DV_HALF;
    int byte_pos = idx % DV_HALF;

    // Load packed byte
    unsigned char packed = V_fp4[idx];

    // Scale group: each scale covers 16 bytes (32 FP4 values)
    int scale_idx = byte_pos / (SCALE_GROUP / 2);  // 16 bytes per scale
    int n_scales = DV / SCALE_GROUP;
    unsigned char scale_e8m0 = V_scale[token * n_scales + scale_idx];

    // E8M0 to float: 2^(scale - 127)
    int exp_bits = ((int)scale_e8m0 - 127 + 127) << 23;
    float scale = __int_as_float(exp_bits);

    // Dequant low nibble → even DV position
    float val_lo = FP4_LUT[packed & 0xFu] * scale;
    V_out[token * DV + 2 * byte_pos] = hip_bfloat16(val_lo);

    // Dequant high nibble → odd DV position
    float val_hi = FP4_LUT[packed >> 4] * scale;
    V_out[token * DV + 2 * byte_pos + 1] = hip_bfloat16(val_hi);
}
