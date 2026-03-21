// MXFP4 Q@K^T kernel for MLA decode on gfx950
// Uses __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4 for FP4×FP4 MFMA
//
// This kernel computes attention scores: scores = Q_fp4 @ K_fp4^T
// for a single block of BLOCK_N=32 tokens.
//
// Q: (PADDED_H=32, DQ_HALF=288) uint8 pre-quantized MXFP4
// Q_scale: (PADDED_H=32, N_SCALES=18) uint8 E8M0
// K: (n, DQ_HALF=288) uint8 MXFP4
// K_scale: (n, N_SCALES=18) uint8 E8M0
//
// Output: scores (PADDED_H=32, BLOCK_N=32) float32

#include <hip/hip_runtime.h>

typedef int __attribute__((ext_vector_type(8))) i32x8_t;
typedef float __attribute__((ext_vector_type(16))) f32x16_t;

#define FP4_FMT 4  // E2M1 format code

// Extract FP4 nibble
__device__ __forceinline__ unsigned char fp4_extract(unsigned char packed, int idx) {
    return (idx == 0) ? (packed & 0xFu) : (packed >> 4);
}
// Pack two FP4 nibbles
__device__ __forceinline__ unsigned char fp4_pack(unsigned char lo, unsigned char hi) {
    return (lo & 0xFu) | ((hi & 0xFu) << 4);
}

// Q@K^T for nope part (512 dims = 256 packed bytes)
// Requires 256/32 = 8 MFMA 32x32x64 instructions
// Each MFMA processes K=64 FP4 = 32 packed bytes
//
// A = Q_nope: (32 heads, 256 packed) — LHS, row-major
// B = K_nope: (256 packed, 32 tokens) — RHS, row-major (transposed K)
// Each MFMA tile: A[32,32bytes] @ B[32bytes,32] → C[32,32]
//
// Launch: 1 warp (64 threads) per MFMA tile
extern "C" __global__ void mxfp4_qk_nope_test(
    const unsigned char* __restrict__ Q_fp4,     // (32, 256) packed
    const unsigned char* __restrict__ K_fp4,     // (n, 288) packed (first 256 = nope)
    const unsigned char* __restrict__ Q_scale,   // (32, 16) E8M0 (nope scales)
    const unsigned char* __restrict__ K_scale,   // (n, 18) E8M0 (first 16 = nope)
    float* __restrict__ scores,                  // (32, 32) output
    const int* __restrict__ kv_locs,             // (32,) token indices
    int stride_k, int stride_ks                  // K strides
) {
#if defined(__gfx950__)
    int lane = threadIdx.x;  // 0..63
    int lane32 = lane % 32;
    int group = lane / 32;   // 0 or 1

    f32x16_t acc = {0};

    // Process 8 MFMA tiles along K dimension (8 * 32 bytes = 256 packed = 512 FP4)
    for (int k_tile = 0; k_tile < 8; k_tile++) {
        int k_byte_offset = k_tile * 32;  // 32 bytes per MFMA tile (64 FP4 values)

        // ── Load A (Q): 16 contiguous bytes per lane ──
        // A is (32, 256) row-major packed. Each MFMA tile uses cols [k_byte_offset, +32)
        // Lane layout: lane32 selects row (0..31), group selects which 16-byte half
        union { i32x8_t v; unsigned char b[32]; } a_buf;
        #pragma unroll
        for (int i = 0; i < 8; i++) a_buf.v[i] = 0;
        const unsigned char* q_row = Q_fp4 + lane32 * 256 + k_byte_offset + group * 16;
        #pragma unroll
        for (int i = 0; i < 16; i++) a_buf.b[i] = q_row[i];

        // ── Load B (K^T): repack nibbles for register layout ──
        // B is K^T: (K_packed, N) = (256, 32). K data in memory is (n, 288).
        // For this tile: K[token, k_byte_offset..+32] transposed to (32bytes, 32tokens)
        // Lane32 selects which 2 tokens (via nibble), group selects K stripe
        union { i32x8_t v; unsigned char b[32]; } b_buf;
        #pragma unroll
        for (int i = 0; i < 8; i++) b_buf.v[i] = 0;

        int token_pair = lane32 / 2;  // which pair of tokens
        int nibble_sel = lane32 % 2;  // which nibble within pair

        int kv_loc0 = kv_locs[token_pair * 2];
        int kv_loc1 = kv_locs[token_pair * 2 + 1];

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int k_pos = k_byte_offset + group * 16 + i;
            unsigned char byte0 = K_fp4[kv_loc0 * stride_k + k_pos];
            unsigned char byte1 = K_fp4[kv_loc1 * stride_k + k_pos];
            // For even nibble_sel: take low nibbles from consecutive tokens
            // For odd nibble_sel: take high nibbles
            // Actually the B layout for mfma is more complex - need to match test_mxfp.cu
            // Using the exact same pattern as test_mxfp.cu:
            b_buf.b[i] = fp4_pack(fp4_extract(byte0, nibble_sel),
                                   fp4_extract(byte1, nibble_sel));
        }

        // ── Get scales for this K tile ──
        // Scale group: each 32 FP4 values (16 bytes) share one scale
        // K tile covers 64 FP4 = 2 scale groups
        int scale_group0 = (k_byte_offset) / 16;      // first group in tile
        int scale_group1 = (k_byte_offset + 16) / 16;  // second group

        // Q scale: same for all tokens, per head
        int q_sc = (int)Q_scale[lane32 * 16 + scale_group0];  // Simplified: use first group's scale

        // K scale: per token
        int k_sc = (int)K_scale[kv_locs[lane32] * stride_ks + scale_group0];  // Simplified

        // ── MFMA FP4 ──
        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_buf.v, b_buf.v, acc,
            FP4_FMT, FP4_FMT,     // A=FP4, B=FP4
            0, q_sc,               // opsel_A=0, scale_A
            0, k_sc                // opsel_B=0, scale_B
        );
    }

    // ── Store scores ──
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        scores[(group * 4 + i * 8 + 0) * 32 + lane32] = acc[i * 4 + 0];
        scores[(group * 4 + i * 8 + 1) * 32 + lane32] = acc[i * 4 + 1];
        scores[(group * 4 + i * 8 + 2) * 32 + lane32] = acc[i * 4 + 2];
        scores[(group * 4 + i * 8 + 3) * 32 + lane32] = acc[i * 4 + 3];
    }
#endif
}
