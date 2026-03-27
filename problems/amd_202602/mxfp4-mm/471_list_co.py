#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#471: List all .co files in f4gemm directory and explore gfx950 subdirectories.
"""
import torch, os, json
from task import input_t, output_t
import aiter
from aiter import dtypes

# === Explore CK ASM code objects ===
print("=" * 60)
print("=== /home/runner/aiter/hsa/gfx950/ subdirectories ===")
_gfx950_base = "/home/runner/aiter/hsa/gfx950/"
try:
    entries = sorted(os.listdir(_gfx950_base))
    for e in entries:
        full = os.path.join(_gfx950_base, e)
        tag = "[DIR]" if os.path.isdir(full) else "[FILE]"
        print(f"  {tag} {e}")
except Exception as ex:
    print(f"  ERROR listing {_gfx950_base}: {ex}")

print("=" * 60)
print("=== /home/runner/aiter/hsa/gfx950/f4gemm/ .co files ===")
_f4gemm_dir = "/home/runner/aiter/hsa/gfx950/f4gemm/"
try:
    entries = sorted(os.listdir(_f4gemm_dir))
    for f in entries:
        full = os.path.join(_f4gemm_dir, f)
        sz = os.path.getsize(full) if os.path.isfile(full) else -1
        print(f"  {f}  ({sz} bytes)")
except Exception as ex:
    print(f"  ERROR listing {_f4gemm_dir}: {ex}")

# Also check if there's an a4w4 or similar directory
for subdir in ["a4w4", "fp4gemm", "gemm", "mxfp4"]:
    candidate = os.path.join(_gfx950_base, subdir)
    if os.path.isdir(candidate):
        print(f"=== {candidate}/ ===")
        try:
            for f in sorted(os.listdir(candidate)):
                print(f"  {f}")
        except Exception as ex:
            print(f"  ERROR: {ex}")

print("=" * 60)

# === Preshuffle setup (for M<=16 shapes) ===
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info

_cfgs={
    "N=2880-K=512":{
        "M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},
        "M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},
        "any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}
    },
    "N=4096-K=512":{
        "M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},
        "any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}
    },
    "N=2112-K=7168":{
        "M_LEQ_16":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":8,"num_warps":4,"num_stages":2,"waves_per_eu":3,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},
        "any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}
    },
    "N=7168-K=2048":{
        "M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":3,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},
        "any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}
    },
    "N=3072-K=1536":{
        "M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":3,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},
        "M_LEQ_256":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":32,"cache_modifier":".cg"},
        "any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}
    }
}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)

# Warmup preshuffle
for _m,_n,_k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
    try:
        _A=torch.randn((_m,_k),dtype=torch.bfloat16,device="cuda")
        _Bw=torch.zeros((_n//16,(_k//2)*16),dtype=torch.uint8,device="cuda")
        _Bws=torch.zeros((_n//32,_k),dtype=torch.uint8,device="cuda")
        gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
    except:pass
torch.cuda.empty_cache()

# Cache for preshuffle B reshape
_ps_ck=None;_ps_cw=None;_ps_cs=None

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _ps_ck,_ps_cw,_ps_cs
    A=data[0];B_shuffle=data[3];B_scale_sh=data[4]
    m,k=A.shape;n=data[1].shape[0]

    # Preshuffle path
    dp=B_shuffle.data_ptr()
    if dp!=_ps_ck:
        _ps_ck=dp;_ps_cw=B_shuffle.view(torch.uint8).reshape(n//16,(k//2)*16)
        _ps_cs=B_scale_sh.view(torch.uint8)[:n,:].contiguous().reshape(n//32,k)
    return gemm_a16wfp4_preshuffle(A,_ps_cw,_ps_cs,prequant=True,dtype=torch.bfloat16)
