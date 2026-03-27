#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#327: Find the CK ASM kernel argument layout by probing the aiter source.
Look at the hsa/ directory for kernel launch code.
"""
import torch, os, subprocess, glob
from task import input_t, output_t

# Find aiter source for kernel launch
print("=== Aiter source search ===")
for pattern in ["/home/runner/aiter/hsa/*.py",
                "/home/runner/aiter/hsa/*.cpp",
                "/home/runner/aiter/hsa/*.hip",
                "/home/runner/aiter/aiter/jit/build/module_gemm_a4w4_asm/*.cpp",
                "/home/runner/aiter/aiter/ops/*.py"]:
    files = glob.glob(pattern)
    for f in sorted(files)[:5]:
        print(f"  {f}")

# Check the hsa directory structure
print("\n=== hsa directory ===")
try:
    r = subprocess.run(["find", "/home/runner/aiter/hsa", "-name", "*.py", "-maxdepth", "2"],
                       capture_output=True, text=True, timeout=5)
    for line in sorted(r.stdout.strip().split('\n'))[:20]:
        print(f"  {line}")
except: pass

# Read the kernel launch code
print("\n=== codegen.py (kernel launch) ===")
try:
    with open("/home/runner/aiter/hsa/codegen.py", "r") as f:
        content = f.read()
    # Find f4gemm related sections
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'f4gemm' in line.lower() or 'BpreShuffle' in line or 'launch' in line.lower():
            start = max(0, i-2)
            end = min(len(lines), i+3)
            for j in range(start, end):
                print(f"  {j}: {lines[j]}")
            print()
except Exception as e:
    print(f"Error: {e}")

# Check for kernel argument struct definitions
print("\n=== Kernel args search ===")
try:
    r = subprocess.run(["grep", "-r", "kernarg\|KernelArg\|launch_kernel\|hipModuleLaunch\|f4gemm_arg",
                        "/home/runner/aiter/hsa/", "--include=*.py"],
                       capture_output=True, text=True, timeout=10)
    for line in r.stdout.strip().split('\n')[:30]:
        print(f"  {line}")
except: pass

# Read the gemm_a4w4 Python wrapper
print("\n=== gemm_a4w4 wrapper ===")
try:
    # Find the Python file that wraps gemm_a4w4
    r = subprocess.run(["grep", "-rl", "gemm_a4w4_asm\|f4gemm_bf16",
                        "/home/runner/aiter/aiter/", "--include=*.py"],
                       capture_output=True, text=True, timeout=10)
    for f in r.stdout.strip().split('\n')[:10]:
        print(f"  File: {f}")
        try:
            with open(f, 'r') as fh:
                content = fh.read()
            if 'f4gemm' in content or 'gemm_a4w4_asm' in content:
                # Find relevant sections
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'f4gemm' in line or 'kernarg' in line.lower() or 'launch' in line.lower():
                        start = max(0, i-1)
                        end = min(len(lines), i+5)
                        for j in range(start, end):
                            print(f"    {j}: {lines[j]}")
                        print()
        except: pass
except: pass

# Check codegen.py for the argument packing code
print("\n=== codegen.py full search ===")
try:
    with open("/home/runner/aiter/hsa/codegen.py", "r") as f:
        content = f.read()
    if 'struct.pack' in content or 'ctypes' in content:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'struct' in line or 'ctypes' in line or 'pack' in line:
                print(f"  {i}: {line}")
    # Also check for hipModuleLaunchKernel
    for i, line in enumerate(content.split('\n')):
        if 'hipModule' in line or 'hipLaunch' in line or 'kernarg' in line.lower():
            print(f"  {i}: {line}")
except: pass

# Standard kernel fallback
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
import json
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
_cfgs = {"N=2880-K=512": {"M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=4096-K=512": {"M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=2112-K=7168": {"M_LEQ_16": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 8, "num_warps": 4, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=7168-K=2048": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=3072-K=1536": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}}
try: _dev = arch_info.get_arch()
except: _dev = "gfx950"
_cd = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
os.makedirs(_cd, exist_ok=True)
for _sk, _cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json", "w") as f:
        json.dump(_cfg, f)

for _m,_n,_k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
    try:
        _A=torch.randn((_m,_k),dtype=torch.bfloat16,device="cuda")
        _Bw=torch.zeros((_n//16,(_k//2)*16),dtype=torch.uint8,device="cuda")
        _Bws=torch.zeros((_n//32,_k),dtype=torch.uint8,device="cuda")
        gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
    except: pass
torch.cuda.empty_cache()

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
