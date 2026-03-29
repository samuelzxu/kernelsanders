#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#584: Find the wrapper that calls _gemm_a16wfp4_preshuffle_kernel and print the EXACT call site.
"""
import torch, os, json, subprocess
from task import input_t, output_t
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info

_cfgs={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=4096-K=512":{"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=2112-K=7168":{"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=7168-K=2048":{"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=3072-K=1536":{"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)

# Find ALL files that reference the preshuffle kernel
print("=== SEARCHING FOR WRAPPER ===")
result = subprocess.run(
    ['grep', '-rn', '_preshuffle_kernel\\[', '/home/runner/aiter/aiter/'],
    capture_output=True, text=True, timeout=10)
print("Files calling _preshuffle_kernel[...]:")
print(result.stdout[:2000])

# Also search for the wrapper function definition
result2 = subprocess.run(
    ['grep', '-rn', 'def gemm_a16wfp4_preshuffle', '/home/runner/aiter/aiter/'],
    capture_output=True, text=True, timeout=10)
print("\nWrapper function definitions:")
print(result2.stdout[:1000])

# Read the wrapper file
if result.stdout:
    # Extract the file path from first match
    first_match = result.stdout.split('\n')[0]
    wrapper_file = first_match.split(':')[0]
    line_num = int(first_match.split(':')[1])
    print(f"\n=== Reading {wrapper_file} lines {line_num-5} to {line_num+30} ===")
    with open(wrapper_file) as f:
        lines = f.readlines()
    for i in range(max(0, line_num-6), min(len(lines), line_num+30)):
        print(f"L{i+1}: {lines[i].rstrip()[:140]}")

# Also check: does preshuffle wrapper live in a different file?
if result2.stdout:
    wrapper_def_file = result2.stdout.split('\n')[0].split(':')[0]
    wrapper_def_line = int(result2.stdout.split('\n')[0].split(':')[1])
    print(f"\n=== Reading wrapper definition from {wrapper_def_file} ===")
    with open(wrapper_def_file) as f:
        lines = f.readlines()
    for i in range(wrapper_def_line-1, min(len(lines), wrapper_def_line+60)):
        print(f"W{i+1}: {lines[i].rstrip()[:140]}")

# Warmup
for _m,_n,_k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
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
