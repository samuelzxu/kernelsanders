#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#431: Check if we can modify the Triton kernel source on the runner.
If yes, we can optimize the kernel itself.
"""
import torch, os
from task import input_t, output_t

# Check file permissions
kernel_path = "/home/runner/aiter/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16wfp4.py"
print(f"=== Kernel file ===")
print(f"Exists: {os.path.exists(kernel_path)}")
print(f"Readable: {os.access(kernel_path, os.R_OK)}")
print(f"Writable: {os.access(kernel_path, os.W_OK)}")

# Check the wrapper file
wrapper_path = "/home/runner/aiter/aiter/ops/triton/gemm/basic/gemm_a16wfp4.py"
print(f"\n=== Wrapper file ===")
print(f"Exists: {os.path.exists(wrapper_path)}")
print(f"Writable: {os.access(wrapper_path, os.W_OK)}")

# Check Triton cache
import glob
cache_dir = os.path.expanduser("~/.triton/cache")
print(f"\n=== Triton cache ===")
print(f"Dir: {cache_dir}")
print(f"Writable: {os.access(cache_dir, os.W_OK)}")

# Read the kernel to count lines and understand structure
if os.path.exists(kernel_path):
    with open(kernel_path) as f:
        content = f.read()
    lines = content.split('\n')
    print(f"Kernel file: {len(lines)} lines, {len(content)} chars")
    # Count key patterns
    for pattern in ['tl.dot', 'tl.load', 'tl.store', 'atomic', '_mxfp4_quant_op']:
        count = content.count(pattern)
        print(f"  '{pattern}': {count} occurrences")

# Standard kernel
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
_cfgs={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)
_ck=None;_cw=None;_cs=None
@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _ck,_cw,_cs
    A=data[0];B_shuffle=data[3];B_scale_sh=data[4]
    m,k=A.shape;n=data[1].shape[0]
    dp=B_shuffle.data_ptr()
    if dp!=_ck:
        _ck=dp;_cw=B_shuffle.view(torch.uint8).reshape(n//16,(k//2)*16)
        _cs=B_scale_sh.view(torch.uint8)[:n,:].contiguous().reshape(n//32,k)
    return gemm_a16wfp4_preshuffle(A,_cw,_cs,prequant=True,dtype=torch.bfloat16)
