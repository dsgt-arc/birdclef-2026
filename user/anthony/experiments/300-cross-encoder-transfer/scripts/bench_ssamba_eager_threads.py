import os, sys, time
from pathlib import Path
import numpy as np, torch
REPO=Path(os.environ["REPO_DIR"])
for p in (REPO/"user/anthony/experiments/004-ssamba-hf-port/ssamba_hf", REPO/"user/anthony/src"):
    sys.path.insert(0, str(p))
from birdclef.encoders.ssamba_hf import SSAMBAHFEncoder
CKPT=os.path.expanduser("~/scratch/birdclef-2026/checkpoints/ssamba/ssamba_small_400.pth")
T=int(os.environ.get("THREADS","4"))
torch.set_num_threads(T)
print(f"torch threads={torch.get_num_threads()}  os.cpu_count={os.cpu_count()}  SLURM_CPUS={os.environ.get('SLURM_CPUS_PER_TASK')}")
enc=SSAMBAHFEncoder(CKPT, model_size="small", input_fdim=128, input_tdim=1024, device="cpu")
for m in enc.model.modules():
    if hasattr(m,"_use_cuda_kernels"): m._use_cuda_kernels=False
mel=torch.randn(16,1024,128)
with torch.no_grad():
    enc.model.forward(mel[:2])  # warmup
    for bs in (1,4,16):
        t0=time.perf_counter()
        for _ in range(2): enc.model.forward(mel[:bs])
        dt=(time.perf_counter()-t0)/2
        print(f"batch{bs:>2}: {dt*1000:8.1f} ms/call -> {bs/dt:6.3f} win/s")
