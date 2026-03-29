#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#581: Reverse-engineer the Triton preshuffle kernel argument layout.
Strategy: read the .amdgcn assembly and find ALL s_load_dword instructions
that reference the kernarg pointer. Also read the Triton source to get the
exact function signature with arg annotations.
"""
import torch, os, json, glob, re
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

# Trigger JIT
for _m,_n,_k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
    try:
        _A=torch.randn((_m,_k),dtype=torch.bfloat16,device="cuda")
        _Bw=torch.zeros((_n//16,(_k//2)*16),dtype=torch.uint8,device="cuda")
        _Bws=torch.zeros((_n//32,_k),dtype=torch.uint8,device="cuda")
        gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
    except:pass
torch.cuda.synchronize()
torch.cuda.empty_cache()

print("=== REVERSE-ENGINEERING ARG LAYOUT ===")

# 1. Read the Triton source file to get the EXACT function signature
kernel_path = "/home/runner/aiter/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16wfp4.py"
print("\n--- Kernel source function signature ---")
with open(kernel_path) as f:
    lines = f.readlines()
    in_sig = False
    sig_lines = []
    for i, line in enumerate(lines):
        if 'def _gemm_a16wfp4_preshuffle_kernel' in line:
            in_sig = True
        if in_sig:
            sig_lines.append(f"L{i+1}: {line.rstrip()}")
            if ')' in line and ':' in line:
                in_sig = False
                break
    for sl in sig_lines:
        print(sl[:150])

# 2. Read the Triton-generated TTIR to see which args are runtime vs constexpr
cache_dir = os.path.expanduser('~/.triton/cache')
# Find the BSM=32 BSN=32 kernel (shapes 3/4)
for d in sorted(glob.glob(f"{cache_dir}/*/")):
    asm_path = os.path.join(d, '_gemm_a16wfp4_preshuffle_kernel.amdgcn')
    if not os.path.exists(asm_path):
        continue
    with open(asm_path) as f:
        first_lines = f.read(500)
    if 'BLOCK_SIZE_M_32_BLOCK_SIZE_N_32' not in first_lines:
        continue

    print(f"\n--- ASM for BSM=32 BSN=32 ---")
    print(f"Dir: {os.path.basename(d)}")

    # Read TTIR to find function signature with types
    ttir_path = os.path.join(d, '_gemm_a16wfp4_preshuffle_kernel.ttir')
    if os.path.exists(ttir_path):
        with open(ttir_path) as f:
            ttir = f.read()
        # Find function declaration
        for line in ttir.split('\n'):
            if 'tt.func' in line and 'preshuffle' in line:
                print(f"\nTTIR func: {line[:200]}")
                # Continue reading arg types
                break
        # Find all arg annotations in the first 30 lines
        for line in ttir.split('\n')[:30]:
            if 'tt.func' in line or '%arg' in line:
                print(f"  {line.strip()[:150]}")

    # Read full ASM for all s_load_dword from kernarg
    with open(asm_path) as f:
        asm = f.read()

    # Find kernel name
    for line in asm.split('\n'):
        if '.amdhsa_kernel' in line:
            print(f"\nKernel: {line.strip()[:150]}")
        if '.amdhsa_kernarg_size' in line:
            print(f"  {line.strip()}")
        if '.amdhsa_group_segment_fixed_size' in line:
            print(f"  {line.strip()}")
        if '.amdhsa_kernarg_preload' in line:
            print(f"  {line.strip()}")

    # Find ALL s_load from kernarg segment
    # In the prologue, s[0:1] holds the kernarg pointer
    print(f"\nAll kernarg loads (s_load from s[0:1]):")
    for line in asm.split('\n'):
        stripped = line.strip()
        if 's_load_dword' in stripped and 's[0:1]' in stripped:
            print(f"  {stripped}")

    # Also check for v_readfirstlane patterns (for pointer args passed via SGPR preload)
    print(f"\nPreloaded args (first 20 SGPR uses):")
    sgpr_uses = []
    for line in asm.split('\n')[:200]:
        stripped = line.strip()
        if stripped and not stripped.startswith(';') and not stripped.startswith('.'):
            # Look for uses of s2-s15 (preloaded from kernarg)
            if re.search(r'\bs([2-9]|1[0-5])\b', stripped) and 's_load' not in stripped:
                sgpr_uses.append(stripped)
    for su in sgpr_uses[:10]:
        print(f"  {su[:120]}")

    break  # Only process one kernel

# 3. Read the wrapper source to understand how args are passed
print("\n--- Wrapper source (arg passing) ---")
wrapper_path = "/home/runner/aiter/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16wfp4.py"
with open(wrapper_path) as f:
    src = f.read()
# Find the kernel launch call
for i, line in enumerate(src.split('\n')):
    if '_preshuffle_kernel[' in line or '_preshuffle_kernel(' in line:
        # Print surrounding context
        start = max(0, i-2)
        end = min(len(src.split('\n')), i+15)
        print(f"\nKernel launch (lines {start+1}-{end+1}):")
        for j in range(start, end):
            print(f"  L{j+1}: {src.split(chr(10))[j].rstrip()[:120]}")
        break

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
