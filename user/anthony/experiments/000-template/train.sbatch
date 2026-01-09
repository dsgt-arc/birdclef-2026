#!/bin/bash
#SBATCH --job-name=000-template
#SBATCH --output=artifacts/slurm-%j.out
#SBATCH --error=artifacts/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu

# Exit on error
set -e

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Navigate to user directory
cd /path/to/user/anthony

# Activate environment (adjust as needed)
# source .venv/bin/activate
# or with uv:
# uv sync

# Run training
uv run python -m birdclef.train \
    --config experiments/000-template/config.yaml \
    --output-dir experiments/000-template/artifacts

echo "End time: $(date)"
