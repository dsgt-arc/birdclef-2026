#!/bin/bash
#SBATCH --job-name=debug-mamba
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -t30 -qembers
#SBATCH --output=logs/debug-mamba-%j.out
set -eu

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

echo "Debug mamba-ssm CUDA kernels"
echo "Job ID: $SLURM_JOB_ID"
hostname
date
nvidia-smi

# Same setup as the main script
scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
user_dir=$repo_dir/user/anthony

export UV_PROJECT_ENVIRONMENT=${TMPDIR}/${SLURM_JOB_ID}/.venv
export UV_LINK_MODE=copy
export HF_HOME=$scratch_dir/.cache/huggingface

# Load CUDA module BEFORE uv sync
module load cuda/12.1.1
echo ""
echo "=== CUDA Environment ==="
echo "CUDA_HOME: $CUDA_HOME"
echo "nvcc version:"
nvcc --version
echo ""

export MAX_JOBS=4

# Show verbose output during mamba-ssm build
cd $user_dir
echo "=== Installing with verbose output ==="
uv sync --extra cuda -v 2>&1 | tee /tmp/uv-sync-$SLURM_JOB_ID.log | tail -100

echo ""
echo "=== Running diagnostic script ==="
${UV_PROJECT_ENVIRONMENT}/bin/python $repo_dir/user/anthony/experiments/001-encoder-benchmark/sbatch/debug-mamba-cuda.py

echo ""
echo "Done!"
date
