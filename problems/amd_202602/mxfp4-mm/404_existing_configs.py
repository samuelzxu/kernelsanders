#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#404: Check what configs ALREADY EXIST on the runner before we overwrite them.
Maybe the runner has pre-tuned configs that are better!
"""
import torch, os, json, glob
from task import input_t, output_t
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info

try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm"

print(f"=== Config dir: {_cd} ===")
print(f"Exists: {os.path.exists(_cd)}")

if os.path.exists(_cd):
    files = sorted(glob.glob(f"{_cd}/*.json"))
    print(f"Config files: {len(files)}")
    for f in files:
        print(f"\n  {os.path.basename(f)}:")
        try:
            with open(f) as fh:
                data = json.load(fh)
            for k,v in data.items():
                print(f"    {k}: {json.dumps(v)[:150]}")
        except: pass
else:
    print("Dir does not exist yet")

# Also check the config path itself
print(f"\n=== AITER_TRITON_CONFIGS_PATH: {AITER_TRITON_CONFIGS_PATH} ===")
for d in sorted(glob.glob(f"{AITER_TRITON_CONFIGS_PATH}/*")):
    print(f"  {os.path.basename(d)}/")
    for f in sorted(glob.glob(f"{d}/*.json"))[:3]:
        print(f"    {os.path.basename(f)}")

# Standard kernel
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
os.makedirs(_cd, exist_ok=True)
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
