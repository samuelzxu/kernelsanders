// Complete fused quant+FP4 GEMM kernel for MI355X (gfx950)
// Reads bf16 A, preshuffle B, shuffled B_scale
// Inline A quantization + MFMA in a single kernel
// 8-wave ping-pong scheduling, LDS XOR swizzle, buffer_load_lds
//
// Target: M=256 N=3072 K=1536 (shape 6) — replace fused_quant + CK_ASM approach
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// ============================================================
// Architecture constants
// ============================================================
#define TILE_M   32        // M rows per block
#define TILE_N   128       // N cols per block (4 warps along N, each 32)
#define K_STEP   128       // K bytes per iteration (256 FP4 values)
#define NWARPS   8
#define NTHREADS (64 * NWARPS)  // 512
#define WARPS_M  2         // ping-pong groups
#define WARPS_N  4         // N-subdivision per group
#define SUB_M    (TILE_M / WARPS_M)   // 16 rows per warp (but MFMA is 32x32...)

// Actually for 32x32 MFMA with 8 warps on a 32x128 tile:
// Each warp computes 32x32. We need 4 sub-tiles along N (128/32=4).
// With 8 warps: 2 groups of 4, each group covers all 4 N sub-tiles.
// Both groups compute the SAME 32x128 output — they handle different K-ranges!
// Wait no — in ping-pong, both groups compute the same tile, same K range.
// The ping-pong is about SCHEDULING, not work division.
//
// Correct design: 8 warps, but only 4 compute unique sub-tiles.
// Warps 0-3 and 4-7 compute the SAME 4 sub-tiles.
// On each SIMD, warp i and warp i+4 alternate compute/memory.
// Both accumulate into the same accumulator (they ARE the same wave, just
// the scheduling interleaves them).
//
// Actually no — each warp has its own registers. Warps 0 and 4 are on the
// same SIMD but have separate accumulators. Both compute ALL sub-tiles.
//
// The HK 8-wave GEMM uses a 256x256 tile with 2x4 warp grid:
// warp_m = wid/4 (0-1), warp_n = wid%4 (0-3)
// Each warp computes 128x64 of the 256x256 output.
// All 8 warps have unique work. The ping-pong just staggers them.
//
// For our 32x128 tile:
// We need a minimum of 4 unique sub-tiles (32x32 each along N).
// With 8 warps: use 1x4 unique sub-tiles, but 2 warp_m groups.
// warp_m=0 warps compute output rows 0-31 (only 32 rows total)
// warp_m=1 warps compute... the same rows? That's redundant.
//
// Better: use a 64x128 tile. Then:
// warp_m=0 → rows 0-31, warp_m=1 → rows 32-63
// warp_n=0..3 → cols 0-31, 32-63, 64-95, 96-127
// 8 unique sub-tiles. Grid: (M/64)*(N/128) blocks.
// For M=256 N=3072: 4*24 = 96 blocks. Low but OK for single-kernel.
//
// For M=16 (shape 2): 16/64 = 0.25 — doesn't divide! Need TILE_M=16.
//
// Let me use a flexible design: TILE_M=32 for small M, with 4 warps computing
// and 4 warps as ping-pong partners. This means each warp pair shares work:
// warp 0 and warp 4 both compute sub-tile (0, 0..31).
// They accumulate separately, then we'd need to sum — BAD.
//
// OK, simplest correct approach for 32x128 tile with 8 warps:
// All 8 warps compute the same 4 sub-tiles, with each warp doing ALL 4.
// Ping-pong just means warps 0-3 are staggered vs 4-7.
// Each warp iterates over all 4 N-sub-tiles per K-iteration.
// This way, no redundant computation per sub-tile.
//
// Wait, that's how the HK 8-wave kernel works for the 256x256 case:
// Each warp has its OWN sub-tile. 8 warps = 8 unique sub-tiles.
//
// For 32x128: we only have 4 unique 32x32 sub-tiles.
// So we need either: (a) 4 warps (no 8-wave), or (b) larger tile.
//
// Let me just use 4 warps with the improved scheduling from the blog
// (setprio + sched_barrier + buffer_load_lds + proper swizzle).
// The occupancy of 2 blocks/CU gives natural 2-wave-per-SIMD ping-pong.

#undef NWARPS
#undef NTHREADS
#undef WARPS_M
#undef WARPS_N

#define TILE_M_K 32
#define TILE_N_K 128
#define K_STEP_K 128      // bytes per K-iteration
#define NWARPS_K 4
#define NTHREADS_K (64 * NWARPS_K)  // 256
#define N_SUBTILES 4      // 128/32 = 4 sub-tiles along N

// LDS swizzle from AMD CDNA4 blog
__device__ __forceinline__ int swizzle_col(int row, int col) {
    const int pair = (row >> 1) & 7;
    const int perm = pair ^ (((pair >> 1) ^ (pair >> 2)) & 1);
    const int mask = perm << 4;
    return col ^ mask;
}

// Flat LDS offset with swizzle (for 16x128 byte subtiles)
__device__ __forceinline__ int lds_offset(int row, int col) {
    return row * K_STEP_K + swizzle_col(row, col);
}

// buffer_load_lds intrinsic
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
using as3_uint32_ptr = uint32_t __attribute__((address_space(3)))*;

extern "C" __device__ void llvm_amdgcn_raw_buffer_load_lds(
    i32x4 rsrc, as3_uint32_ptr lds_ptr, int size,
    int voffset, int soffset, int offset, int aux)
    __asm("llvm.amdgcn.raw.buffer.load.lds");

struct buffer_resource {
    uint64_t ptr;
    uint32_t range;
    uint32_t config;
};

__device__ __forceinline__ i32x4 make_srsrc(const void* ptr, uint32_t range_bytes) {
    buffer_resource rsrc = {reinterpret_cast<uint64_t>(ptr), range_bytes, 0x110000};
    return *reinterpret_cast<const i32x4*>(&rsrc);
}

// ============================================================
// Inline A quantization: bf16 → FP4 + E8M0 scale
// Processes one group of 32 bf16 values
// ============================================================
__device__ __forceinline__ void quant_group_fp4(
    const unsigned short* __restrict__ a_bf16,  // 32 bf16 values
    unsigned char* __restrict__ a_fp4_out,       // 16 packed FP4 bytes
    unsigned char& scale_out                     // E8M0 scale
) {
    float vals[32];
    float amax = 0.0f;

    // Load 32 bf16 values via uint4 (vectorized)
    const uint4* r4 = (const uint4*)a_bf16;
    #pragma unroll
    for (int v = 0; v < 4; v++) {
        uint4 chunk = r4[v];
        unsigned int w[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float lo = __uint_as_float((w[j] & 0xFFFFu) << 16);
            float hi = __uint_as_float(w[j] & 0xFFFF0000u);
            vals[v*8+j*2] = lo;
            vals[v*8+j*2+1] = hi;
            amax = fmaxf(amax, fmaxf(fabsf(lo), fabsf(hi)));
        }
    }

    // Compute E8M0 scale
    unsigned int ai = __float_as_uint(amax);
    unsigned int ar = (ai + 0x200000u) & 0xFF800000u;
    int eb = (ar >> 23) & 0xFF;
    int su = eb - 129;
    int sb = su + 127;
    if (sb < 0) sb = 0;
    scale_out = (unsigned char)sb;

    // Compute quantization factor
    int qe = 127 - su;
    if (qe < 1) qe = 0;
    if (qe > 254) qe = 254;
    float qs = __uint_as_float((unsigned int)qe << 23);
    if (amax == 0.0f) qs = 0.0f;

    // FP4 quantization constants
    const unsigned int dmi = 149u << 23;
    float dmf = __uint_as_float(dmi);
    const int vta = ((int)(1-127) << 23) + (1 << 21) - 1;

    auto fp4 = [&](float v) -> unsigned char {
        float qf = v * qs;
        unsigned int qx = __float_as_uint(qf);
        unsigned int s = qx & 0x80000000u;
        qx ^= s;
        float qp = __uint_as_float(qx);
        unsigned char r;
        if (qp >= 6.0f) r = 0x7;
        else if (qp < 1.0f) r = (unsigned char)((__float_as_uint(qp + dmf) - dmi) & 0xFF);
        else {
            unsigned int mo = (qx >> 22) & 1;
            r = (unsigned char)((((unsigned int)((int)qx + vta) + mo) >> 22) & 0xFF);
        }
        return (r & 0x7) | ((unsigned char)(s >> 28) & 0x8);
    };

    // Pack 32 FP4 values into 16 bytes
    #pragma unroll
    for (int i = 0; i < 16; i++)
        a_fp4_out[i] = fp4(vals[2*i]) | (fp4(vals[2*i+1]) << 4);
}

// ============================================================
// Main kernel: fused A-quant + preshuffle-B FP4 GEMM
// ============================================================
extern "C" __global__ __launch_bounds__(NTHREADS_K, 2)
void fp4_fused_gemm(
    const unsigned short* __restrict__ A_bf16,   // [M, K] bf16
    const unsigned char*  __restrict__ B_preshuf, // [N/16, Kh*16] preshuffle FP4
    const unsigned char*  __restrict__ B_scale_sh,// [N/32, K] shuffled E8M0
    unsigned short*       __restrict__ C,         // [M, N] bf16 output
    int M, int N, int K                           // K in bf16 elements
) {
    // LDS for A (quantized FP4) and B (preshuffle FP4)
    // A: TILE_M × K_STEP = 32 × 128 = 4096 bytes per buffer
    // B: TILE_N × K_STEP = 128 × 128 = 16384 bytes per buffer
    // Double-buffered: 2 × (4096 + 16384) = 40960 bytes
    // Plus A_scale: 32 × 8 × 2 = 512 bytes (scale groups per K_STEP)
    __shared__ unsigned char lds_a[2][TILE_M_K * K_STEP_K];     // 8KB
    __shared__ unsigned char lds_b[2][TILE_N_K * K_STEP_K];     // 32KB
    __shared__ unsigned char lds_as[2][TILE_M_K * (K_STEP_K/16)]; // scale: 32*8*2=512B

    const int Kh = K / 2;             // K in packed FP4 bytes
    const int K_groups = K / 32;      // number of scale groups
    const int K_iters = Kh / K_STEP_K; // K iterations

    // Block assignment with XCD swizzle
    int bid = blockIdx.x;
    const int npm = M / TILE_M_K;
    const int npn = N / TILE_N_K;
    const int NW = gridDim.x;
    const int NXCD = 8;
    if (NW >= NXCD) {
        int remapped = (bid % NXCD) * ((NW + NXCD - 1) / NXCD) + (bid / NXCD);
        if (remapped < NW) bid = remapped;
    }
    // L2-friendly tile ordering
    const int WGM = (npm < 8) ? npm : 8;
    const int nwig = WGM * npn;
    const int gid = bid / nwig;
    const int fpm = gid * WGM;
    const int gsm = min(npm - fpm, WGM);
    const int pm = fpm + ((bid % nwig) % gsm);
    const int pn = (bid % nwig) / gsm;

    const int tid = threadIdx.x;
    const int wid = tid / 64;
    const int lane = tid % 64;
    const int lane32 = lane & 31;
    const int group = lane >> 5;

    // Each of 4 warps handles one 32x32 N-sub-tile
    const int warp_n = wid;  // 0-3, selects which 32-col sub-tile
    const int a_row_base = pm * TILE_M_K;
    const int b_col_base = pn * TILE_N_K + warp_n * 32;

    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int v8i;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float v16f;
    v16f c_acc = {};  // 32x32 accumulator

    // ============================================================
    // Phase 1: Quantize A for first K-tile and load to LDS
    // Each thread quantizes one group of 32 bf16 values
    // ============================================================
    // A has TILE_M * K_groups total scale groups for the full K dimension.
    // Per K-tile (K_STEP bytes = K_STEP*2 bf16 = K_STEP*2/32 = 8 groups):
    // Total groups per tile: TILE_M * 8 = 256. With 256 threads: 1 group/thread.

    auto load_a_tile = [&](int buf, int k_iter) {
        // Quantize A for this K-tile: 32 rows × 8 groups = 256 groups
        int groups_per_iter = TILE_M_K * (K_STEP_K / 16);  // 32 * 8 = 256
        if (tid < groups_per_iter) {
            int row = tid / (K_STEP_K / 16);    // which M row
            int grp = tid % (K_STEP_K / 16);    // which group within this K-tile
            int abs_row = a_row_base + row;
            int abs_grp = k_iter * (K_STEP_K / 16) + grp;

            unsigned char fp4_bytes[16];
            unsigned char scale_val;

            if (abs_row < M && abs_grp < K_groups) {
                quant_group_fp4(
                    A_bf16 + abs_row * K + abs_grp * 32,
                    fp4_bytes, scale_val);
            } else {
                for (int i = 0; i < 16; i++) fp4_bytes[i] = 0;
                scale_val = 0;
            }

            // Write FP4 to LDS with swizzle
            int lds_row = row;
            int lds_col_base = grp * 16;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int col = lds_col_base + i;
                lds_a[buf][lds_offset(lds_row, col)] = fp4_bytes[i];
            }
            // Write scale
            lds_as[buf][row * (K_STEP_K/16) + grp] = scale_val;
        }
    };

    auto load_b_tile = [&](int buf, int k_iter) {
        // B preshuffle: (N/16, Kh*16) — load TILE_N × K_STEP bytes
        // 128 rows × 128 bytes = 16384 bytes. 256 threads → 64 bytes/thread = 4 rounds of 16B
        int bw_stride = Kh * 16;  // bytes per super-row in preshuffle format
        int k_byte_start = k_iter * K_STEP_K;

        for (int round = 0; round < 4; round++) {
            int flat = tid + round * NTHREADS_K;  // 0..1023
            int b_row = flat / (K_STEP_K / 16);   // which N-row within tile (0..127)
            int b_grp = flat % (K_STEP_K / 16);   // which 16-byte group (0..7)

            int n_col = pn * TILE_N_K + b_row;
            int k_byte = k_byte_start + b_grp * 16;

            if (n_col < N && k_byte + 15 < Kh) {
                // Preshuffle address formula
                int super_row = n_col / 16;
                int n_within = n_col % 16;
                int kb = k_byte / 32;
                int kh_half = (k_byte % 32) / 16;
                int ki = k_byte % 16;
                // Load 16 contiguous bytes from preshuffle
                // In preshuffle format, bytes for one (n_within, k_block) are at:
                // base + kb*512 + kh_half*256 + n_within*16
                const unsigned char* src = B_preshuf + super_row * bw_stride
                    + kb * 512 + kh_half * 256 + n_within * 16;

                int lds_row = b_row;
                int lds_col = b_grp * 16;
                // Copy 16 bytes with swizzle
                // Since ki=0 for aligned loads, we can do uint4 copy
                int swz = lds_offset(lds_row, lds_col);
                *(uint4*)&lds_b[buf][swz] = *(const uint4*)src;
            } else {
                int lds_row = b_row;
                int lds_col = b_grp * 16;
                int swz = lds_offset(lds_row, lds_col);
                *(uint4*)&lds_b[buf][swz] = make_uint4(0,0,0,0);
            }
        }
    };

    // Load first K-tile
    load_a_tile(0, 0);
    load_b_tile(0, 0);
    __syncthreads();

    int tic = 0, toc = 1;

    for (int ki = 0; ki < K_iters; ki++, tic ^= 1, toc ^= 1) {
        // Prefetch next K-tile
        if (ki + 1 < K_iters) {
            load_a_tile(toc, ki + 1);
            load_b_tile(toc, ki + 1);
        }

        // Load B_scale for this K-tile (from shuffled format)
        int scale_base_grp = ki * (K_STEP_K / 16);  // 8 groups per K-tile
        unsigned char b_scales[8];
        {
            int n_col = b_col_base + lane32;
            if (n_col < N) {
                int kg8 = K_groups / 8;
                int n0 = n_col / 32;
                int n1 = (n_col & 31) / 16;
                int n2 = n_col & 15;
                #pragma unroll
                for (int s = 0; s < 8; s++) {
                    int g = scale_base_grp + s;
                    int g0 = g / 8;
                    int g1 = (g & 7) / 4;
                    int g2 = g & 3;
                    int idx = n0 * (kg8 * 256) + g0 * 256 + g2 * 64 + n2 * 4 + g1 * 2 + n1;
                    b_scales[s] = B_scale_sh[idx];
                }
            } else {
                #pragma unroll
                for (int s = 0; s < 8; s++) b_scales[s] = 0;
            }
        }

        // Load A_scale from LDS
        unsigned char a_scales[8];
        {
            int a_row = lane32;  // row within tile
            if (a_row_base + a_row < M) {
                #pragma unroll
                for (int s = 0; s < 8; s++)
                    a_scales[s] = lds_as[tic][a_row * (K_STEP_K/16) + s];
            } else {
                #pragma unroll
                for (int s = 0; s < 8; s++) a_scales[s] = 0;
            }
        }

        // Wait for LDS writes to complete
        asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)");
        __syncthreads();

        // 4 sub-MFMA iterations per K-tile (128 bytes = 4 × 32 bytes)
        __builtin_amdgcn_s_setprio(1);
        #pragma unroll
        for (int sub = 0; sub < 4; sub++) {
            int k_byte_off = sub * 32 + group * 16;

            // Read A from swizzled LDS
            int a_swz = lds_offset(lane32, k_byte_off);
            v8i a_reg = {};
            *(uint4*)&a_reg = *(uint4*)&lds_a[tic][a_swz];

            // Read B from swizzled LDS
            int b_lds_row = warp_n * 32 + lane32;
            int b_swz = lds_offset(b_lds_row, k_byte_off);
            v8i b_reg = {};
            *(uint4*)&b_reg = *(uint4*)&lds_b[tic][b_swz];

            // Pack scales: v3 (broadcast own to both bytes)
            unsigned char a_sc = a_scales[sub * 2 + group];
            unsigned char b_sc = b_scales[sub * 2 + group];
            unsigned int pas = (unsigned int)a_sc | ((unsigned int)a_sc << 8);
            unsigned int pbs = (unsigned int)b_sc | ((unsigned int)b_sc << 8);

            __builtin_amdgcn_sched_barrier(0);
            c_acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a_reg, b_reg, c_acc, 4, 4, 0, pas, 0, pbs);
            __builtin_amdgcn_sched_barrier(0);
        }
        __builtin_amdgcn_s_setprio(0);
        __syncthreads();
    }

    // Store output as bf16
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int r = a_row_base + group * 4 + i * 8 + j;
            int c = b_col_base + lane32;
            if (r < M && c < N) {
                float val = c_acc[i * 4 + j];
                unsigned int u = __float_as_uint(val);
                u = u + (((u >> 16) & 1) + 0x7FFF);
                C[r * N + c] = (unsigned short)(u >> 16);
            }
        }
    }
}

// ============================================================
// PyTorch wrapper
// ============================================================
torch::Tensor run_fused_gemm(
    torch::Tensor A_bf16,      // [M, K] bf16
    torch::Tensor B_preshuf,   // preshuffle FP4
    torch::Tensor B_scale_sh,  // shuffled E8M0
    int64_t N_val
) {
    int M = (int)A_bf16.size(0);
    int K = (int)A_bf16.size(1);

    auto C = torch::empty({M, (int)N_val}, torch::dtype(torch::kBFloat16).device(A_bf16.device()));

    int grid = (M / TILE_M_K) * ((int)N_val / TILE_N_K);
    fp4_fused_gemm<<<grid, NTHREADS_K, 0, 0>>>(
        (const unsigned short*)A_bf16.data_ptr(),
        (const unsigned char*)B_preshuf.data_ptr(),
        (const unsigned char*)B_scale_sh.data_ptr(),
        (unsigned short*)C.data_ptr(),
        M, (int)N_val, K
    );
    return C;
}

TORCH_LIBRARY(fp4_fused, m) {
    m.def("run", &run_fused_gemm);
}
