#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#595: Read the Triton-generated assembly to understand its A quant strategy.
"""
import torch, os, json, glob
from task import input_t, output_t
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info

_cfgs={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=4096-K=512":{"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=2112-K=7168":{"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=7168-K=2048":{"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=3072-K=1536":{"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)

for _m,_n,_k in [(32,4096,512)]:
    _A=torch.randn((_m,_k),dtype=torch.bfloat16,device="cuda")
    _Bw=torch.zeros((_n//16,(_k//2)*16),dtype=torch.uint8,device="cuda")
    _Bws=torch.zeros((_n//32,_k),dtype=torch.uint8,device="cuda")
    gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
torch.cuda.synchronize()

# Find the BSM=32 BSN=32 assembly
cache = os.path.expanduser('~/.triton/cache')
for asm_path in sorted(glob.glob(f"{cache}/**/*preshuffle*.amdgcn", recursive=True)):
    with open(asm_path) as f:
        content = f.read()
    if 'BLOCK_SIZE_M_32_BLOCK_SIZE_N_32' in content:
        lines = content.split('\n')
        print(f"=== ASM for BSM=32 BSN=32 ({len(lines)} lines) ===")

        # Count instruction types
        mfma_count = sum(1 for l in lines if 'v_mfma' in l)
        global_load = sum(1 for l in lines if 'global_load' in l or 'buffer_load' in l)
        ds_read = sum(1 for l in lines if 'ds_read' in l)
        ds_write = sum(1 for l in lines if 'ds_write' in l)
        valu = sum(1 for l in lines if l.strip().startswith('v_') and 'mfma' not in l)
        salu = sum(1 for l in lines if l.strip().startswith('s_') and 'waitcnt' not in l and 'barrier' not in l)
        barriers = sum(1 for l in lines if 's_barrier' in l or 's_waitcnt' in l)

        print(f"  MFMA calls: {mfma_count}")
        print(f"  Global loads: {global_load}")
        print(f"  DS reads: {ds_read}")
        print(f"  DS writes: {ds_write}")
        print(f"  VALU ops: {valu}")
        print(f"  SALU ops: {salu}")
        print(f"  Barriers: {barriers}")

        # Show the MFMA calls and surrounding context
        print(f"\n  === MFMA instructions ===")
        for i, line in enumerate(lines):
            if 'v_mfma' in line:
                start = max(0, i-2)
                end = min(len(lines), i+2)
                for j in range(start, end):
                    print(f"  L{j}: {lines[j].strip()[:120]}")
                print()

        # Show the first 50 lines of the kernel body (after .text)
        in_body = False
        body_lines = 0
        print(f"\n  === First 30 lines of kernel body ===")
        for i, line in enumerate(lines):
            if '.text' in line:
                in_body = True
                continue
            if in_body and line.strip() and not line.strip().startswith('.') and not line.strip().startswith(';'):
                print(f"  L{i}: {line.strip()[:120]}")
                body_lines += 1
                if body_lines >= 30:
                    break

        # Check: does it use buffer_load_lds?
        has_buf_load_lds = any('buffer_load_lds' in l for l in lines)
        print(f"\n  Uses buffer_load_lds: {has_buf_load_lds}")

        # Check: how does it load A? Look for A-related loads
        print(f"\n  === A loading pattern ===")
        for i, line in enumerate(lines):
            if 'global_load' in line.lower() and i < len(lines) - 1:
                print(f"  L{i}: {line.strip()[:120]}")
                if body_lines > 100:
                    break

        break

for _m,_n,_k in [(4,2880,512),(16,2112,7168),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
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
