#!/bin/bash
#SBATCH --job-name=ssamba-mamba-setup
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=6 --mem=32G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -t60 -qinferno
#SBATCH --output=logs/mamba-setup-%j.out

# Run this ONCE to install mamba-ssm CUDA kernels, then submit the array job:
#   SETUP_JOB=$(sbatch --parsable sbatch/setup-mamba.sbatch)
#   ARRAY_JOB=$(sbatch --parsable --dependency=afterok:$SETUP_JOB sbatch/run-benchmark.sbatch)
#   sbatch --dependency=afterany:$ARRAY_JOB sbatch/collect-results.sbatch

set -eu

module load cuda/12.1.1

echo "=== SSAMBA mamba-ssm CUDA Setup ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date

repo_dir=~/arc/birdclef-2026
project_dir=$repo_dir/user/anthony
venv=$project_dir/.venv

cd $project_dir

echo ""
echo "=== PyTorch / CUDA info ==="
$venv/bin/python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=== Install pre-built mamba-ssm + causal-conv1d wheels ==="
# Uses [tool.uv.sources] in pyproject.toml which pins pre-built wheels
# compiled for torch2.5+cu12+cp311 from GitHub releases.
# --all-packages ensures ssamba-hf workspace member is also installed.
uv sync --all-packages --extra cuda

echo ""
echo "=== Verify CUDA kernels ==="
$venv/bin/python -c "
from ssamba_hf.mamba import is_cuda_kernel_available
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'PyTorch: {torch.__version__}')
print(f'mamba-ssm CUDA kernels: {is_cuda_kernel_available()}')
if is_cuda_kernel_available():
    print('SUCCESS: CUDA kernels ready')
else:
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    except Exception as e:
        print(f'Import error: {e}')
    exit(1)
"

echo ""
echo "=== Setup Complete ==="
date
