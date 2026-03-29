#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#576: Deep probe of Triton cache — read HSACO metadata, arg layouts, kernel names.
Goal: understand how to load + launch Triton-compiled kernels via hipModuleLaunchKernel.
"""
import torch, os, json, glob
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

print("=== DEEP PROBE ===")

# 1. List ALL files in each cache directory
cache_dir = os.path.expanduser('~/.triton/cache')
hsaco_dirs = set()
for hsaco in glob.glob(f"{cache_dir}/**/*.hsaco", recursive=True):
    hsaco_dirs.add(os.path.dirname(hsaco))

for d in sorted(hsaco_dirs):
    print(f"\n--- {os.path.basename(d)} ---")
    for f in sorted(os.listdir(d)):
        fpath = os.path.join(d, f)
        sz = os.path.getsize(fpath)
        print(f"  {f} ({sz}B)")
        # Read metadata files
        if f.endswith('.json'):
            try:
                with open(fpath) as jf:
                    meta = json.load(jf)
                print(f"    METADATA: {json.dumps(meta, indent=None)[:500]}")
            except:
                with open(fpath) as jf:
                    print(f"    RAW: {jf.read()[:500]}")
        elif f.endswith('.py') or f.endswith('.txt') or f.endswith('.ir'):
            with open(fpath) as tf:
                content = tf.read()[:300]
                print(f"    CONTENT: {content}")

# 2. Try triton.compiler.CompiledKernel
print("\n=== CompiledKernel inspection ===")
try:
    from triton.compiler import CompiledKernel
    print(f"CompiledKernel attrs: {[a for a in dir(CompiledKernel) if not a.startswith('_')]}")
except Exception as e:
    print(f"CompiledKernel: {e}")

# 3. Try to access the kernel through triton.runtime
print("\n=== Triton runtime ===")
try:
    import triton
    print(f"triton.runtime attrs: {[a for a in dir(triton.runtime) if not a.startswith('_')][:20]}")
    if hasattr(triton.runtime, 'driver'):
        print(f"triton.runtime.driver: {dir(triton.runtime.driver)[:10]}")
except Exception as e:
    print(f"triton.runtime: {e}")

# 4. Check the actual kernel module file on runner
print("\n=== Kernel module source ===")
kernel_path = "/home/runner/aiter/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16wfp4.py"
try:
    with open(kernel_path) as kf:
        lines = kf.readlines()
    print(f"File has {len(lines)} lines")
    # Find kernel function definitions
    for i, line in enumerate(lines):
        if 'def _gemm' in line or '@gluon' in line or '@triton' in line:
            print(f"  Line {i+1}: {line.rstrip()[:120]}")
except Exception as e:
    print(f"Can't read kernel: {e}")

# 5. Check the wrapper function — does it expose the compiled kernel?
print("\n=== Wrapper function internals ===")
try:
    import inspect
    src = inspect.getsource(gemm_a16wfp4_preshuffle)
    # Look for references to the JIT kernel
    for line in src.split('\n'):
        if 'kernel' in line.lower() or 'grid' in line.lower() or 'launch' in line.lower():
            print(f"  {line.strip()[:120]}")
except Exception as e:
    print(f"Source inspection: {e}")

# Fallback kernel
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
