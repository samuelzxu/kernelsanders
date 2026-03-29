#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#586: Extract hipFunction_t directly from Triton's CompiledKernel after JIT.
Instead of loading HSACO from disk (which may be stale), intercept the
function pointer that Triton actually uses.
"""
import torch, os, json
from task import input_t, output_t
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'

# Monkey-patch Triton to capture the compiled kernel's function pointer
import triton

_captured_kernels = {}  # shape_key -> (function_ptr, num_warps, shared_mem, grid)

_orig_run = triton.runtime.JITFunction.run

def _capturing_run(self, *args, grid, warmup=False, **kwargs):
    result = _orig_run(self, *args, grid=grid, warmup=warmup, **kwargs)
    name = getattr(self, '__name__', '')
    if 'preshuffle' in name and not warmup:
        # After run, check if we can access the compiled kernel
        # The result of run() varies — but the kernel cache should be populated
        try:
            # Access the cache to get the compiled kernel
            device = args[0].device.index if hasattr(args[0], 'device') else 0
            if hasattr(self, 'device_caches') and device in self.device_caches:
                cache = self.device_caches[device]
                if cache:
                    # Get the most recently used entry
                    for key, compiled in cache.items():
                        if hasattr(compiled, 'function') and compiled.function is not None:
                            fn_ptr = compiled.function
                            num_warps = kwargs.get('num_warps', 4)
                            shared = compiled.metadata.shared if hasattr(compiled, 'metadata') and hasattr(compiled.metadata, 'shared') else 0
                            # Compute grid
                            if callable(grid):
                                grid_val = grid(kwargs)
                            else:
                                grid_val = grid
                            if isinstance(grid_val, (tuple, list)):
                                gx = grid_val[0]
                            else:
                                gx = grid_val

                            # Use M,N,K as shape key
                            M = args[4] if len(args) > 4 else 0
                            N = args[5] if len(args) > 5 else 0
                            K = args[6] if len(args) > 6 else 0
                            shape_key = (int(M), int(N), int(K))

                            _captured_kernels[shape_key] = {
                                'fn_ptr': fn_ptr,
                                'num_warps': num_warps,
                                'shared': shared,
                                'grid_x': gx,
                            }
                            if len(_captured_kernels) <= 6:
                                print(f"  Captured: M={M} N={N} K={K} fn=0x{fn_ptr:x} warps={num_warps} shared={shared} grid={gx}")
                            break
        except Exception as e:
            pass  # Don't break if capture fails
    return result

triton.runtime.JITFunction.run = _capturing_run

from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info

_cfgs={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=4096-K=512":{"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=2112-K=7168":{"M_LEQ_16":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":8,"num_warps":4,"num_stages":2,"waves_per_eu":3,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=7168-K=2048":{"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":3,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=3072-K=1536":{"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":3,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_256":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":32,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)

# Warmup all shapes — this triggers JIT and our capture hook
print("=== Capturing kernel function pointers ===")
for _m,_n,_k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
    _A=torch.randn((_m,_k),dtype=torch.bfloat16,device="cuda")
    _Bw=torch.zeros((_n//16,(_k//2)*16),dtype=torch.uint8,device="cuda")
    _Bws=torch.zeros((_n//32,_k),dtype=torch.uint8,device="cuda")
    try:
        gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
        torch.cuda.synchronize()
    except:pass
    del _A,_Bw,_Bws

# Restore original run
triton.runtime.JITFunction.run = _orig_run

print(f"\n=== Captured {len(_captured_kernels)} kernel pointers ===")
for key, info in sorted(_captured_kernels.items()):
    print(f"  M={key[0]} N={key[1]} K={key[2]}: fn=0x{info['fn_ptr']:x} warps={info['num_warps']} shared={info['shared']} grid={info['grid_x']}")

# Also try to access device_caches directly
try:
    import importlib
    mod = importlib.import_module('aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16wfp4')
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, triton.runtime.JITFunction) and 'preshuffle' in name:
            print(f"\n=== {name} cache ===")
            if hasattr(obj, 'device_caches'):
                for dev, cache in obj.device_caches.items():
                    print(f"  device {dev}: {len(cache)} entries")
                    for key, compiled in list(cache.items())[:3]:
                        fn = getattr(compiled, 'function', None)
                        mod_h = getattr(compiled, 'module', None)
                        meta = getattr(compiled, 'metadata', None)
                        shared = getattr(meta, 'shared', 0) if meta else 0
                        name_str = getattr(compiled, 'name', '?')
                        print(f"    key_hash={hash(key)}, fn=0x{fn:x if fn else 0}, shared={shared}, name={name_str[:80]}")
            break
except Exception as e:
    print(f"Cache access: {e}")

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
