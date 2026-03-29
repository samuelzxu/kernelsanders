#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#577: Extract Triton kernel argument layout from .amdgcn assembly.
Then attempt to load HSACO and launch via hipModuleLaunchKernel with known args.
"""
import torch, os, json, glob, ctypes
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

# Trigger JIT for shape 1 (simplest, KSPLIT=1)
_A=torch.randn((4,512),dtype=torch.bfloat16,device="cuda")
_Bw=torch.zeros((2880//16,(512//2)*16),dtype=torch.uint8,device="cuda")
_Bws=torch.zeros((2880//32,512),dtype=torch.uint8,device="cuda")
_ref=gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
torch.cuda.synchronize()

print("=== ARG LAYOUT PROBE ===")

# 1. Find the HSACO for shape 1 (warps=4, waves=1, stages=1)
cache_dir = os.path.expanduser('~/.triton/cache')
target_hsaco = None
for d in glob.glob(f"{cache_dir}/*/"):
    meta_file = os.path.join(d, '_gemm_a16wfp4_preshuffle_kernel.json')
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            meta = json.load(f)
        if meta.get('num_warps') == 4 and meta.get('waves_per_eu') == 1 and meta.get('num_stages') == 1:
            target_hsaco = os.path.join(d, '_gemm_a16wfp4_preshuffle_kernel.hsaco')
            target_asm = os.path.join(d, '_gemm_a16wfp4_preshuffle_kernel.amdgcn')
            print(f"Found shape1 HSACO: {target_hsaco} ({os.path.getsize(target_hsaco)}B)")
            print(f"Found shape1 ASM: {target_asm} ({os.path.getsize(target_asm)}B)")
            break

# 2. Read the amdgcn assembly to find kernarg_segment_size and argument layout
if target_asm and os.path.exists(target_asm):
    with open(target_asm) as f:
        asm = f.read()
    # Find kernel metadata
    for line in asm.split('\n'):
        if 'kernarg_segment_size' in line or '.amdhsa_kernel' in line or 'kernarg' in line.lower():
            print(f"ASM: {line.strip()[:120]}")
    # Find argument loading instructions (s_load_dword* from kernarg)
    arg_loads = [l.strip() for l in asm.split('\n') if 's_load_dword' in l and 'kernarg' in l.lower()]
    if not arg_loads:
        # Try finding them by offset pattern
        arg_loads = [l.strip() for l in asm.split('\n') if 's_load_dword' in l][:20]
    print(f"\nFirst 20 arg loads:")
    for l in arg_loads[:20]:
        print(f"  {l[:120]}")

# 3. Read the .source file to get the Triton source with parameter annotations
source_file = target_hsaco.replace('.hsaco', '.source') if target_hsaco else None
if source_file and os.path.exists(source_file):
    with open(source_file) as f:
        src = f.read()
    # Find function signature with types
    for line in src.split('\n')[:50]:
        if 'def ' in line or 'tl.constexpr' in line or 'pointer' in line.lower():
            print(f"SRC: {line.strip()[:120]}")

# 4. Try to load the HSACO via hipModuleLoad
if target_hsaco:
    print(f"\n=== Loading HSACO via hipModuleLoad ===")
    try:
        hip = ctypes.CDLL("libamdhip64.so")

        # hipModuleLoad
        module = ctypes.c_void_p()
        ret = hip.hipModuleLoad(ctypes.byref(module), target_hsaco.encode())
        print(f"hipModuleLoad: {ret} (0=success), module={module.value}")

        if ret == 0:
            # hipModuleGetFunction
            func = ctypes.c_void_p()
            # Triton kernel name in HSACO
            ret2 = hip.hipModuleGetFunction(ctypes.byref(func),
                module, b"_gemm_a16wfp4_preshuffle_kernel")
            print(f"hipModuleGetFunction('_gemm_a16wfp4_preshuffle_kernel'): {ret2}, func={func.value}")

            if ret2 != 0:
                # Try with mangled name
                ret3 = hip.hipModuleGetFunction(ctypes.byref(func),
                    module, b"_gemm_a16wfp4_preshuffle_kernel_0d1d2d3d4d5d6d7d8d9d10d11d12d13d14d")
                print(f"Mangled name attempt: {ret3}")

            # List all functions in module
            # Can't easily enumerate, but try common patterns
            for suffix in ['', '_0d1d2d3d', '_0d1d2d3d4d5d6d7d8d9d10d11d12d13d14d',
                          '_0d1d2d3d4d5d6d7d8d9d10d11d12d13d14d15d16d17d']:
                name = f"_gemm_a16wfp4_preshuffle_kernel{suffix}"
                ret_try = hip.hipModuleGetFunction(ctypes.byref(func), module, name.encode())
                if ret_try == 0:
                    print(f"FOUND: {name} -> func={func.value}")
                    break
    except Exception as e:
        print(f"HSACO load failed: {e}")

# 5. Check triton.runtime.JITFunction internals
print("\n=== JITFunction internals ===")
try:
    import triton
    # Access the actual kernel function
    import importlib
    mod = importlib.import_module('aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16wfp4')
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, triton.runtime.JITFunction):
            print(f"JITFunction: {name}")
            print(f"  cache keys: {list(obj.cache.keys())[:3] if hasattr(obj, 'cache') else 'N/A'}")
            if hasattr(obj, 'cache') and obj.cache:
                for key, compiled in list(obj.cache.items())[:1]:
                    print(f"  key type: {type(key)}")
                    print(f"  compiled type: {type(compiled)}")
                    print(f"  compiled attrs: {[a for a in dir(compiled) if not a.startswith('_')]}")
                    if hasattr(compiled, 'asm'):
                        print(f"  .asm keys: {list(compiled.asm.keys()) if isinstance(compiled.asm, dict) else type(compiled.asm)}")
                    if hasattr(compiled, 'function'):
                        print(f"  .function: {compiled.function}")
                    if hasattr(compiled, 'metadata'):
                        meta = compiled.metadata
                        print(f"  .metadata type: {type(meta)}")
                        if hasattr(meta, '__dict__'):
                            print(f"  .metadata keys: {list(meta.__dict__.keys())[:10]}")
except Exception as e:
    print(f"JITFunction probe: {e}")

# Warmup remaining shapes
for _m,_n,_k in [(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
    try:
        _A=torch.randn((_m,_k),dtype=torch.bfloat16,device="cuda")
        _Bw=torch.zeros((_n//16,(_k//2)*16),dtype=torch.uint8,device="cuda")
        _Bws=torch.zeros((_n//32,_k),dtype=torch.uint8,device="cuda")
        gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
    except:pass
torch.cuda.empty_cache()

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
