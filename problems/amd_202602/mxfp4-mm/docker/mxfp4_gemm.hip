// Custom MXFP4 FP4×FP4 GEMM kernel for MI355X (gfx950)
// Uses __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4 with FP4 (Atype=4, Btype=4)
// Fuses BF16→FP4 quantization of A into the GEMM kernel (like preshuffle)
//
// Layout: A [M,K] bf16, B [N,K/2] FP4 packed (preshuffle), output [M,N] bf16
// Scales: A_scale computed inline, B_scale [N,K/32] E8M0 (shuffled)
//
// Tile: 32x32 output, K=64 per MFMA instruction
// Each wavefront (64 threads) computes one 32x32 tile

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// BF16 to float conversion
__device__ __forceinline__ float bf16_to_float(unsigned short b) {
    return __uint_as_float((unsigned int)b << 16);
}

// Float to BF16 conversion (round to nearest even)
__device__ __forceinline__ unsigned short float_to_bf16(float f) {
    unsigned int u = __float_as_uint(f);
    // Round to nearest even
    unsigned int rounding_bias = ((u >> 16) & 1) + 0x7FFF;
    u += rounding_bias;
    return (unsigned short)(u >> 16);
}

// MXFP4 quantization of 32 bf16 values
// Returns: 16 packed FP4 bytes in out_fp4[16], E8M0 scale in *out_scale
__device__ __forceinline__ void quant_group_32(
    const unsigned short* __restrict__ a_bf16,  // 32 bf16 values
    unsigned char* out_fp4,  // 16 packed bytes
    unsigned char* out_scale  // 1 E8M0 byte
) {
    // Step 1: Find max absolute value
    float vals[32];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        vals[i] = bf16_to_float(a_bf16[i]);
        amax = fmaxf(amax, fabsf(vals[i]));
    }

    // Step 2: Compute E8M0 scale
    unsigned int ai = __float_as_uint(amax);
    unsigned int amax_rounded = (ai + 0x200000u) & 0xFF800000u;
    int exp_bits = (amax_rounded >> 23) & 0xFF;
    int scale_unbiased = exp_bits - 129;  // exp - 127 - 2
    int scale_biased = scale_unbiased + 127;
    if (scale_biased < 0) scale_biased = 0;
    *out_scale = (unsigned char)scale_biased;

    // Step 3: Compute quant scale = 2^(-scale_unbiased)
    int qs_exp = 127 - scale_unbiased;
    if (qs_exp < 1) qs_exp = 0;
    if (qs_exp > 254) qs_exp = 254;
    float quant_scale = __uint_as_float((unsigned int)qs_exp << 23);
    if (amax == 0.0f) quant_scale = 0.0f;

    // Step 4: Quantize each value to FP4 E2M1
    const float max_normal = 6.0f;
    const float min_normal = 1.0f;
    const unsigned int denorm_mask_int = 149u << 23;
    float denorm_mask_float = __uint_as_float(denorm_mask_int);
    const int val_to_add = ((int)(1 - 127) << 23) + (1 << 21) - 1;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        auto to_fp4 = [&](float v) -> unsigned char {
            float qx_f = v * quant_scale;
            unsigned int qx = __float_as_uint(qx_f);
            unsigned int s = qx & 0x80000000u;
            qx ^= s;
            float qx_pos = __uint_as_float(qx);
            unsigned char result;
            if (qx_pos >= max_normal) {
                result = 0x7;
            } else if (qx_pos < min_normal) {
                result = (unsigned char)((__float_as_uint(qx_pos + denorm_mask_float) - denorm_mask_int) & 0xFF);
            } else {
                unsigned int mo = (qx >> 22) & 1;
                result = (unsigned char)((((unsigned int)((int)qx + val_to_add) + mo) >> 22) & 0xFF);
            }
            return (result & 0x7) | ((unsigned char)(s >> 28) & 0x8);
        };
        out_fp4[i] = to_fp4(vals[2*i]) | (to_fp4(vals[2*i+1]) << 4);
    }
}

// Main GEMM kernel: C[M,N] = quant(A[M,K]) @ B_preshuffle[N,K/2]
// Each block computes one 32x32 output tile
// Grid: (ceil(N/32), ceil(M/32), 1)
extern "C" __global__ void mxfp4_fused_gemm(
    const unsigned short* __restrict__ A,      // [M, K] bf16 as uint16
    const unsigned char* __restrict__ B,        // [N, K/2] FP4 packed (preshuffle)
    const unsigned char* __restrict__ B_scale,  // [N_pad, K/32] E8M0 (shuffled)
    unsigned short* __restrict__ C,             // [M, N] bf16 output
    int M, int N, int K,
    int stride_a,      // A row stride (= K)
    int stride_b,      // B row stride (= K/2)
    int stride_bs,     // B_scale row stride (= K/32)
    int stride_c       // C row stride (= N)
) {
    int tile_n = blockIdx.x * 32;
    int tile_m = blockIdx.y * 32;

    int lane = threadIdx.x;  // 0-63 within wavefront
    int lane32 = lane & 31;  // lane within half-warp
    int group = lane >> 5;   // 0 or 1 (which half of wavefront)

    // Output accumulator: 16 FP32 values per lane
    float acc[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;

    // K loop: process 64 FP4 elements per iteration
    int K_groups = K / 32;  // number of scale groups
    int K_iters = K / 64;   // number of MFMA iterations

    for (int ki = 0; ki < K_iters; ki++) {
        int k_base = ki * 64;

        // === Load and quantize A tile: 32 rows × 64 columns ===
        // Each lane handles specific rows based on MFMA layout
        // Lane32 = row index, group = K-half (0..31 or 32..63)
        unsigned char a_fp4[16];  // 32 FP4 values = 16 bytes
        unsigned char a_scale;

        int a_row = tile_m + lane32;
        int a_k_start = k_base + group * 32;

        if (a_row < M) {
            const unsigned short* a_ptr = A + a_row * stride_a + a_k_start;
            quant_group_32(a_ptr, a_fp4, &a_scale);
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) a_fp4[i] = 0;
            a_scale = 127;  // scale = 1.0 (neutral)
        }

        // === Load B tile: 64 rows × 32 columns ===
        // B is preshuffle format: [N, K/2] packed FP4
        // For MFMA FP4: lane32 maps to N-column, group maps to K-half
        unsigned char b_fp4[16];  // 32 FP4 values = 16 bytes
        unsigned char b_scale;

        int b_col = tile_n + lane32;
        int b_k_start = k_base + group * 32;

        if (b_col < N) {
            // Load 16 bytes of packed FP4 from B
            const unsigned char* b_ptr = B + b_col * stride_b + b_k_start / 2;
            #pragma unroll
            for (int i = 0; i < 16; i++) b_fp4[i] = b_ptr[i];

            // Load B scale for this group
            int scale_idx = (k_base / 32) + group;
            b_scale = B_scale[b_col * stride_bs + scale_idx];
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) b_fp4[i] = 0;
            b_scale = 127;
        }

        // === Pack scales for MFMA ===
        // MFMA expects: scale_a = scale_group0 | (scale_group1 << 8)
        // group0 = lanes 0-31, group1 = lanes 32-63
        unsigned int other_a_scale = __shfl_xor((unsigned int)a_scale, 32);
        unsigned int other_b_scale = __shfl_xor((unsigned int)b_scale, 32);
        unsigned int packed_a_scale, packed_b_scale;
        if (group == 0) {
            packed_a_scale = (unsigned int)a_scale | (other_a_scale << 8);
            packed_b_scale = (unsigned int)b_scale | (other_b_scale << 8);
        } else {
            packed_a_scale = other_a_scale | ((unsigned int)a_scale << 8);
            packed_b_scale = other_b_scale | ((unsigned int)b_scale << 8);
        }

        // === Call MFMA ===
        // Pack FP4 bytes into the register format expected by MFMA
        // a_fp4[16] = 128 bits = 4x int32 for the 256-bit vector (upper 128 bits = 0)
        typedef int v8si __attribute__((ext_vector_type(8)));
        typedef float v16f __attribute__((vector_size(64)));

        v8si a_reg = {};
        a_reg[0] = *(int*)&a_fp4[0];
        a_reg[1] = *(int*)&a_fp4[4];
        a_reg[2] = *(int*)&a_fp4[8];
        a_reg[3] = *(int*)&a_fp4[12];

        v8si b_reg = {};
        b_reg[0] = *(int*)&b_fp4[0];
        b_reg[1] = *(int*)&b_fp4[4];
        b_reg[2] = *(int*)&b_fp4[8];
        b_reg[3] = *(int*)&b_fp4[12];

        v16f c_reg;
        #pragma unroll
        for (int i = 0; i < 16; i++) c_reg[i] = acc[i];

        // FP4 MFMA: 32x32x64, Atype=4 (FP4), Btype=4 (FP4)
        c_reg = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_reg, b_reg, c_reg,
            4, 4,       // Atype=FP4, Btype=FP4
            0, packed_a_scale,  // OPSEL_A=0, scale_a
            0, packed_b_scale   // OPSEL_B=0, scale_b
        );

        #pragma unroll
        for (int i = 0; i < 16; i++) acc[i] = c_reg[i];
    }

    // === Store output ===
    // MFMA 32x32 output layout:
    // acc[i*4+j] → row = group*4 + i*8 + j, col = lane32
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int out_row = tile_m + group * 4 + i * 8 + j;
            int out_col = tile_n + lane32;
            if (out_row < M && out_col < N) {
                C[out_row * stride_c + out_col] = float_to_bf16(acc[i * 4 + j]);
            }
        }
    }
}
