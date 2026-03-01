#!/bin/bash
#SBATCH --job-name=ssamba-cuda-test
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem=32G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -t60 -qinferno
#SBATCH --output=logs/test-cuda-kernels-%j.out

set -eu

# Load CUDA 12.1 to match the venv
module load cuda/12.1.1

echo "=== SSAMBA CUDA Kernel Test ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date

# Paths
repo_dir=~/arc/birdclef-2026
ssamba_hf_dir=$repo_dir/user/anthony/experiments/004-ssamba-hf-port/ssamba_hf

cd $ssamba_hf_dir

echo ""
echo "=== Step 1: Install mamba-ssm (if not present) ==="
# Limit parallel jobs to reduce memory pressure during CUDA kernel compilation
export MAX_JOBS=4
uv pip install mamba-ssm causal-conv1d || echo "mamba-ssm install failed or already installed"

echo ""
echo "=== Step 2: Verify mamba-ssm availability ==="
uv run python -c "
from ssamba_hf import is_cuda_kernel_available, MAMBA_SSM_AVAILABLE
print(f'MAMBA_SSM_AVAILABLE: {MAMBA_SSM_AVAILABLE}')
print(f'is_cuda_kernel_available(): {is_cuda_kernel_available()}')
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=== Step 3: Run pytest tests ==="
uv run --extra dev pytest tests/test_cuda_kernels.py -v --tb=short -x

echo ""
echo "=== Test Complete ==="
date
