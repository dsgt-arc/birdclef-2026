#!/bin/bash
#SBATCH --job-name=ssamba-token-bench
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=6 --mem=32G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -t30 -qinferno
#SBATCH --array=0-10
#SBATCH --output=logs/token-bench-%A_%a.out

# Usage (chain all three jobs):
#   cd user/anthony/experiments/005-ssamba-token-context
#   SETUP_JOB=$(sbatch --parsable sbatch/setup-mamba.sbatch)
#   ARRAY_JOB=$(sbatch --parsable --dependency=afterok:$SETUP_JOB sbatch/run-benchmark.sbatch)
#   sbatch --dependency=afterany:$ARRAY_JOB sbatch/collect-results.sbatch

set -eu

module load cuda/12.1.1

echo "=== SSAMBA Token Benchmark — Config $SLURM_ARRAY_TASK_ID ==="
echo "Array Job: $SLURM_ARRAY_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date

repo_dir=~/arc/birdclef-2026
project_dir=$repo_dir/user/anthony
exp_dir=$repo_dir/user/anthony/experiments/005-ssamba-token-context
PYTHON=$project_dir/.venv/bin/python

# Use $venv/bin/python directly — NOT uv run — because parallel array tasks
# sharing the same venv will race to sync and corrupt each other's packages.

# Verify CUDA kernels (installed by setup-mamba.sbatch)
echo ""
$PYTHON -c "
from ssamba_hf.mamba import is_cuda_kernel_available
assert is_cuda_kernel_available(), 'CUDA kernels not available — run setup-mamba.sbatch first'
print('CUDA Mamba kernels: OK')
"

echo ""
echo "=== Running benchmark ==="
mkdir -p $exp_dir/results $exp_dir/logs

$PYTHON $exp_dir/scripts/benchmark_token_shapes.py \
    --config-index $SLURM_ARRAY_TASK_ID \
    --results-dir $exp_dir/results

echo ""
echo "=== Task $SLURM_ARRAY_TASK_ID Complete ==="
date
