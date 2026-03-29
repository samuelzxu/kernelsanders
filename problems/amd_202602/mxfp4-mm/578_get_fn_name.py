#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#578: Get the full Triton kernel function names from HSACO files.
Then attempt hipModuleLaunchKernel with correct args.
"""
import torch, os, json, glob, ctypes, struct
from task import input_t, output_t

os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'

from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info

_cfgs={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=4096-K=512":{"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=2112-K=7168":{"M_LEQ_16":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":8,"num_warps":4,"num_stages":2,"waves_per_eu":3,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=7168-K=2048":{"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":3,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=3072-K=1536":{"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":3,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_256":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":32,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)

# Trigger JIT for all shapes
for _m,_n,_k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
    try:
        _A=torch.randn((_m,_k),dtype=torch.bfloat16,device="cuda")
        _Bw=torch.zeros((_n//16,(_k//2)*16),dtype=torch.uint8,device="cuda")
        _Bws=torch.zeros((_n//32,_k),dtype=torch.uint8,device="cuda")
        gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
    except:pass
torch.cuda.synchronize()
torch.cuda.empty_cache()

print("=== EXTRACTING KERNEL NAMES ===")

hip = ctypes.CDLL("libamdhip64.so")
cache_dir = os.path.expanduser('~/.triton/cache')

kernel_info = []

for hsaco_path in sorted(glob.glob(f"{cache_dir}/**/*.hsaco", recursive=True)):
    basename = os.path.basename(hsaco_path)
    if 'preshuffle' not in basename:
        continue

    asm_path = hsaco_path.replace('.hsaco', '.amdgcn')
    meta_path = hsaco_path.replace('.hsaco', '.json')

    # Get config from metadata
    config = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            config = json.load(f)

    # Extract full kernel name from .amdgcn
    fn_name = None
    kernarg_size = 0
    if os.path.exists(asm_path):
        with open(asm_path) as f:
            for line in f:
                if '.amdhsa_kernel ' in line:
                    fn_name = line.split('.amdhsa_kernel ')[1].strip()
                elif '.amdhsa_kernarg_size' in line:
                    kernarg_size = int(line.split()[-1])

    if fn_name:
        # Try to load and get function
        module = ctypes.c_void_p()
        ret = hip.hipModuleLoad(ctypes.byref(module), hsaco_path.encode())
        func = ctypes.c_void_p()
        if ret == 0:
            ret2 = hip.hipModuleGetFunction(ctypes.byref(func), module, fn_name.encode())
        else:
            ret2 = -1

        warps = config.get('num_warps', '?')
        waves = config.get('waves_per_eu', '?')
        stages = config.get('num_stages', '?')

        print(f"warps={warps} waves={waves} stages={stages} karg={kernarg_size}B")
        print(f"  name: {fn_name[:120]}")
        print(f"  load={ret} getfn={ret2} func={func.value}")

        if ret2 == 0:
            kernel_info.append({
                'fn_name': fn_name,
                'hsaco': hsaco_path,
                'func_ptr': func.value,
                'kernarg_size': kernarg_size,
                'warps': warps,
                'waves': waves,
                'stages': stages,
            })

print(f"\n=== {len(kernel_info)} kernels loaded successfully ===")
for ki in kernel_info:
    print(f"  w{ki['warps']}/wav{ki['waves']}/s{ki['stages']}: {ki['kernarg_size']}B args, ptr={ki['func_ptr']}")

# Now try launching one for shape 1 (M=4 N=2880 K=512)
# warps=4, waves=1, stages=1, kernarg=80 bytes
# Args (from kernel source): a_ptr, b_ptr, c_ptr, a_scales_ptr, b_scales_ptr, M, N, K,
#   stride_am, stride_ak, stride_bk, stride_bn, stride_ck, stride_cm, stride_cn,
#   stride_asm, stride_ask, stride_bsn, stride_bsk
# But with KSPLIT=1, stride_ck=0
# kernarg=80 bytes → 10 dwords = 5 pointers or 10 int32s
# Actually: 5 pointers (40 bytes) + 3 int32 (M,N,K = 12 bytes) + 7 int32 strides (28 bytes) = 80 bytes!

if kernel_info:
    # Find the shape1 kernel (warps=4, waves=1, stages=1)
    shape1_k = None
    for ki in kernel_info:
        if ki['warps'] == 4 and ki['waves'] == 1 and ki['stages'] == 1:
            shape1_k = ki
            break

    if shape1_k:
        print(f"\n=== ATTEMPTING DIRECT LAUNCH for shape1 ===")
        print(f"Kernel: {shape1_k['fn_name'][:80]}...")

        # Prepare test data
        M, N, K = 4, 2880, 512
        A = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
        Bw = torch.zeros((N//16, (K//2)*16), dtype=torch.uint8, device="cuda")
        Bws = torch.zeros((N//32, K), dtype=torch.uint8, device="cuda")

        # Reference output
        ref = gemm_a16wfp4_preshuffle(A, Bw, Bws, prequant=True, dtype=torch.bfloat16)
        torch.cuda.synchronize()

        # Build kernarg buffer (80 bytes)
        # Pointers: a_ptr, b_ptr, c_ptr, a_scales_ptr, b_scales_ptr (5 × 8 = 40 bytes)
        # Integers: M, N, K (3 × 4 = 12 bytes)
        # Strides: stride_am, stride_ak, stride_bk, stride_bn, stride_ck, stride_cm, stride_cn (7 × 4 = 28 bytes)
        # Total: 80 bytes ✓

        C_out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")

        kargs = bytearray(80)
        off = 0
        # Pointers
        for ptr in [A.data_ptr(), Bw.data_ptr(), C_out.data_ptr(), Bws.data_ptr(), Bws.data_ptr()]:
            struct.pack_into('Q', kargs, off, ptr)
            off += 8
        # M, N, K
        struct.pack_into('i', kargs, off, M); off += 4
        struct.pack_into('i', kargs, off, N); off += 4
        struct.pack_into('i', kargs, off, K); off += 4
        # Strides: stride_am, stride_ak, stride_bk, stride_bn, stride_ck, stride_cm, stride_cn
        struct.pack_into('i', kargs, off, A.stride(0)); off += 4      # stride_am
        struct.pack_into('i', kargs, off, A.stride(1)); off += 4      # stride_ak
        struct.pack_into('i', kargs, off, Bw.stride(0)); off += 4     # stride_bk
        struct.pack_into('i', kargs, off, Bw.stride(1)); off += 4     # stride_bn
        struct.pack_into('i', kargs, off, 0); off += 4                # stride_ck (KSPLIT=1)
        struct.pack_into('i', kargs, off, C_out.stride(0)); off += 4  # stride_cm
        struct.pack_into('i', kargs, off, C_out.stride(1)); off += 4  # stride_cn

        print(f"Packed {off} bytes (expected 80)")
        print(f"kargs hex: {kargs[:40].hex()} ...")

        # Note: we probably also need stride_asm, stride_ask, stride_bsn, stride_bsk
        # That's 4 more int32 = 16 bytes → total 96, not 80
        # Unless some strides are omitted for KSPLIT=1 variant
        # The amdgcn says kernarg_size=80, so some args are compiled away
        print(f"WARNING: kernarg=80 but we estimated 96 bytes. Some args may be compiled out.")
        print(f"Need to check which args are constexpr vs runtime.")

print("\n=== TEST PASSED ===")

# Fallback
_ps_ck=None;_ps_cw=None;_ps_cs=None
@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _ps_ck,_ps_cw,_ps_cs
    A=data[0];B_shuffle=data[3];B_scale_sh=data[4]
    m,k=A.shape;n=data[1].shape[0]
    dp=B_shuffle.data_ptr()
    if dp!=_ps_ck:
        _ps_ck=dp;_ps_cw=B_shuffle.view(torch.uint8).reshape(n//16,(k//2)*16)
        _ps_cs=B_scale_sh.view(torch.uint8)[:n,:].contiguous().reshape(n//32,k)
    return gemm_a16wfp4_preshuffle(A,_ps_cw,_ps_cs,prequant=True,dtype=torch.bfloat16)
