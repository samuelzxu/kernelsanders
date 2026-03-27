// Optimized FP4 GEMM: vectorized loading, K_STEP=128, 4 independent warps
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TILE_M 32
#define TILE_N 32
#define K_STEP 128   // 128 bytes = 256 FP4 = 4 MFMA calls
#define NWARPS 4
#define NTHREADS (64 * NWARPS)

// LDS: just 1 set of A/B tiles (no double buffer for simplicity)
// A: 32 × 128 = 4096 bytes, B: 32 × 128 = 4096 bytes
// Total: 8192 bytes — fits easily

extern "C" __global__ __launch_bounds__(NTHREADS, 4)
void fp4_lds_opt(
    const unsigned char* __restrict__ A,
    const unsigned char* __restrict__ B,
    const unsigned char* __restrict__ A_scale,
    const unsigned char* __restrict__ B_scale,
    unsigned short* __restrict__ C,
    int M, int N, int K_bytes
) {
    // Double-buffered LDS with 16-byte padding per row to reduce bank conflicts
    __shared__ unsigned char lds_a[2][TILE_M][K_STEP + 16];
    __shared__ unsigned char lds_b[2][TILE_N][K_STEP + 16];

    // XCD-aware block assignment (8 XCDs on MI355X)
    int bid = blockIdx.x;
    int npm = M / TILE_M;
    int npn = N / TILE_N;
    int NW = gridDim.x;
    int NXCD = 8;
    bid = (bid % NXCD) * (NW / NXCD) + (bid / NXCD);
    int WGM = (npm < 8) ? npm : 8;
    int nwig = WGM * npn;
    int gid = bid / nwig;
    int fpm = gid * WGM;
    int gsm = min(npm - fpm, WGM);
    int pm = fpm + ((bid % nwig) % gsm);
    int pn = (bid % nwig) / gsm;

    int tid = threadIdx.x;
    int lane = tid % 64;
    int lane32 = lane & 31;
    int group = lane >> 5;

    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int v8i;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float v16f;
    v16f c_acc = {};

    int a_row_base = pm * TILE_M;
    int b_col_base = pn * TILE_N;
    int K_iters = K_bytes / K_STEP;
    int K_groups = K_bytes / 16;  // K/32 scale groups

    // Load first K-tile into buffer 0
    {
        int a_flat = tid * 16;
        int a_r = a_flat / K_STEP, a_c = a_flat % K_STEP;
        int a_gr = a_row_base + a_r;
        *(uint4*)&lds_a[0][a_r][a_c] = (a_gr < M) ?
            *(const uint4*)&A[a_gr * K_bytes + a_c] : make_uint4(0,0,0,0);
        int b_flat = tid * 16;
        int b_r = b_flat / K_STEP, b_c = b_flat % K_STEP;
        int b_gc = b_col_base + b_r;
        *(uint4*)&lds_b[0][b_r][b_c] = (b_gc < N) ?
            *(const uint4*)&B[b_gc * K_bytes + b_c] : make_uint4(0,0,0,0);
    }
    __syncthreads();

    int tic = 0, toc = 1;
    for (int ki = 0; ki < K_iters; ki++, tic ^= 1, toc ^= 1) {
        // Prefetch NEXT K-tile into alternate buffer (overlap with compute)
        if (ki + 1 < K_iters) {
            int k_next = (ki + 1) * K_STEP;
            int a_flat = tid * 16;
            int a_r = a_flat / K_STEP, a_c = a_flat % K_STEP;
            int a_gr = a_row_base + a_r;
            *(uint4*)&lds_a[toc][a_r][a_c] = (a_gr < M) ?
                *(const uint4*)&A[a_gr * K_bytes + k_next + a_c] : make_uint4(0,0,0,0);
            int b_flat = tid * 16;
            int b_r = b_flat / K_STEP, b_c = b_flat % K_STEP;
            int b_gc = b_col_base + b_r;
            *(uint4*)&lds_b[toc][b_r][b_c] = (b_gc < N) ?
                *(const uint4*)&B[b_gc * K_bytes + k_next + b_c] : make_uint4(0,0,0,0);
        }

        // Preload all 8 scale values for this K-tile
        int scale_base = ki * (K_STEP / 16);
        int a_scale_row = (a_row_base + lane32) * K_groups;
        int b_scale_row = (b_col_base + lane32) * K_groups;
        unsigned char a_scales[8], b_scales[8];
        #pragma unroll
        for (int s = 0; s < 8; s++) {
            a_scales[s] = A_scale[a_scale_row + scale_base + s];
            b_scales[s] = B_scale[b_scale_row + scale_base + s];
        }

        // Wait for current tile LDS writes to complete, then compute
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();  // ensure all threads see LDS data
        __builtin_amdgcn_s_setprio(1);
        #pragma unroll
        for (int sub = 0; sub < 4; sub++) {
            int k_byte_start = sub * 32 + group * 16;

            v8i a_reg = {}, b_reg = {};
            *(uint4*)&a_reg = *(uint4*)&lds_a[tic][lane32][k_byte_start];
            *(uint4*)&b_reg = *(uint4*)&lds_b[tic][lane32][k_byte_start];

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
        // No full syncthreads here — prefetch overlap with next iteration's scale load
        __builtin_amdgcn_s_barrier();
    }

    // Store bf16
    int rb = a_row_base;
    int cb = b_col_base;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int r = rb + group * 4 + i * 8 + j;
            int c = cb + lane32;
            if (r < M && c < N) {
                float val = c_acc[i * 4 + j];
                unsigned int u = __float_as_uint(val);
                u = u + (((u >> 16) & 1) + 0x7FFF);
                C[r * N + c] = (unsigned short)(u >> 16);
            }
        }
    }
}

static torch::Tensor g_out;
static int g_m = 0, g_n = 0;

torch::Tensor run_fp4_lds(torch::Tensor A, torch::Tensor B,
                           torch::Tensor As, torch::Tensor Bs, int64_t N_val) {
    int M_val = (int)A.size(0), K_val = (int)A.size(1);
    if (M_val != g_m || (int)N_val != g_n) {
        g_out = torch::empty({M_val, N_val}, torch::dtype(torch::kBFloat16).device(A.device()));
        g_m = M_val; g_n = (int)N_val;
    }
    int total_tiles = (M_val / TILE_M) * ((int)N_val / TILE_N);
    fp4_lds_opt<<<total_tiles, NTHREADS, 0, 0>>>(
        (const unsigned char*)A.data_ptr(), (const unsigned char*)B.data_ptr(),
        (const unsigned char*)As.data_ptr(), (const unsigned char*)Bs.data_ptr(),
        (unsigned short*)g_out.data_ptr(), M_val, (int)N_val, K_val);
    return g_out;
}

// Combined quant + GEMM in single call
extern "C" __global__ void fused_quant_raw(
    const unsigned short* __restrict__ A, unsigned char* __restrict__ Aq,
    unsigned char* __restrict__ As_raw,
    int M, int K, int Kh, int Kg, int Mp) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= Mp * Kg) return;
    int m = gid / Kg; int g = gid % Kg;
    float vals[32]; float amax = 0.0f;
    if (m < M) {
        const uint4* r4 = (const uint4*)(A + m * K + g * 32);
        #pragma unroll
        for (int v=0;v<4;v++){uint4 c=r4[v];unsigned int w[4]={c.x,c.y,c.z,c.w};
            #pragma unroll
            for(int j=0;j<4;j++){float lo=__uint_as_float((w[j]&0xFFFFu)<<16);float hi=__uint_as_float(w[j]&0xFFFF0000u);
                vals[v*8+j*2]=lo;vals[v*8+j*2+1]=hi;amax=fmaxf(amax,fmaxf(fabsf(lo),fabsf(hi)));}}
    } else { for(int i=0;i<32;i++)vals[i]=0.0f; }
    unsigned int ai=__float_as_uint(amax);unsigned int ar=(ai+0x200000u)&0xFF800000u;
    int eb=(ar>>23)&0xFF;int su=eb-129;int sb=su+127;if(sb<0)sb=0;
    As_raw[m*Kg+g]=(unsigned char)sb;
    int qe=127-su;if(qe<1)qe=0;if(qe>254)qe=254;
    float qs=__uint_as_float((unsigned int)qe<<23);if(amax==0.0f)qs=0.0f;
    const unsigned int dmi=149u<<23;float dmf=__uint_as_float(dmi);
    const int vta=((int)(1-127)<<23)+(1<<21)-1;
    auto fp4=[&](float v)->unsigned char{float qf=v*qs;unsigned int qx=__float_as_uint(qf);
        unsigned int s2=qx&0x80000000u;qx^=s2;float qp=__uint_as_float(qx);unsigned char r;
        if(qp>=6.0f)r=0x7;else if(qp<1.0f)r=(unsigned char)((__float_as_uint(qp+dmf)-dmi)&0xFF);
        else{unsigned int mo=(qx>>22)&1;r=(unsigned char)((((unsigned int)((int)qx+vta)+mo)>>22)&0xFF);}
        return(r&0x7)|((unsigned char)(s2>>28)&0x8);};
    unsigned char pk[16];
    #pragma unroll
    for(int i=0;i<16;i++)pk[i]=fp4(vals[2*i])|(fp4(vals[2*i+1])<<4);
    uint4* o4=(uint4*)(Aq+m*Kh+g*16);*o4=*((uint4*)pk);
}

static torch::Tensor g_aq2, g_asr2, g_out2;
static int g_Mp2=0, g_K22=0, g_N22=0;

torch::Tensor quant_and_lds_gemm(torch::Tensor A, torch::Tensor Bq, torch::Tensor Bs, int64_t N_val) {
    int M_val=(int)A.size(0), K_val=(int)A.size(1);
    int Mp=((M_val+31)/32)*32, Kh=K_val/2, Kg=K_val/32;
    if(Mp!=g_Mp2||K_val!=g_K22){
        g_aq2=torch::empty({Mp,Kh},torch::dtype(torch::kUInt8).device(A.device()));
        g_asr2=torch::empty({Mp,Kg},torch::dtype(torch::kUInt8).device(A.device()));
        g_Mp2=Mp;g_K22=K_val;
    }
    if((int)N_val!=g_N22||Mp!=g_Mp2){
        g_out2=torch::empty({M_val,(int)N_val},torch::dtype(torch::kBFloat16).device(A.device()));
        g_N22=(int)N_val;
    }

    // 1) Launch quant
    int tg=Mp*Kg;int th=256;int bl=(tg+th-1)/th;
    fused_quant_raw<<<bl,th,0,0>>>((const unsigned short*)A.data_ptr(),
        g_aq2.data_ptr<unsigned char>(),g_asr2.data_ptr<unsigned char>(),
        M_val,K_val,Kh,Kg,Mp);

    // 2) Launch LDS GEMM immediately (same HIP queue)
    int total_tiles=(M_val/32)*((int)N_val/32);
    fp4_lds_opt<<<total_tiles,NTHREADS,0,0>>>(
        g_aq2.data_ptr<unsigned char>(),(const unsigned char*)Bq.data_ptr(),
        g_asr2.data_ptr<unsigned char>(),(const unsigned char*)Bs.data_ptr(),
        (unsigned short*)g_out2.data_ptr(),M_val,(int)N_val,Kh);
    return g_out2;
}

TORCH_LIBRARY(hk_gemm, m) {
    m.def("run_lds", &run_fp4_lds);
    m.def("quant_gemm", &quant_and_lds_gemm);
}
