#!/bin/bash
#SBATCH --job-name=bench-ast
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -t240 -qembers
#SBATCH --output=logs/benchmark-ast-%j.out
set -eu

echo "Running AST encoder benchmark on ESC-50"
echo "Job ID: $SLURM_JOB_ID"
hostname
date
nvidia-smi

scratch_dir=~/scratch/birdclef-2026
repo_dir=~/arc/birdclef-2026
gpu_stats_file=logs/gpu_stats-ast-${SLURM_JOB_ID}.csv

# Start GPU monitoring in background
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv -l 5 > "$gpu_stats_file" &
GPU_MONITOR_PID=$!
trap "kill $GPU_MONITOR_PID 2>/dev/null || true" EXIT

# Use scratch venv
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv

# Set HuggingFace cache to scratch
export HF_HOME=$scratch_dir/.cache/huggingface

cd $repo_dir

# Sync dependencies and install the birdclef package
uv sync
uv pip install -e user/anthony

# Run AST benchmark (uses HuggingFace - auto-downloads)
uv run python -m birdclef.benchmark \
    --encoder ast \
    --esc50-root $repo_dir/vendor/ESC-50 \
    --checkpoint-dir $scratch_dir/checkpoints \
    --output-dir $scratch_dir/outputs/esc50-benchmark-partc \
    --batch-size 32 \
    --epochs 100 \
    --lr 1e-3

echo ""
echo "Done!"
echo "GPU stats saved to: $gpu_stats_file"
date
