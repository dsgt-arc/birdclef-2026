#!/bin/bash
#SBATCH --job-name=infer-bench-gpu
#SBATCH --account=paceship-dsgt_clef2025
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -t120 -qembers
#SBATCH --output=logs/benchmark-gpu-%j.out
set -eu

echo "Running inference benchmark (GPU)"
echo "Job ID: $SLURM_JOB_ID"
hostname
date
nvidia-smi

scratch_dir=~/scratch/birdclef-2026
repo_dir=~/arc/birdclef-2026

source $scratch_dir/.venv/bin/activate
# uv sync --active

cd $repo_dir
# uv pip install -e user/adrian
# uv pip install -e user/anthony

gpu_stats_file=logs/gpu_stats-${SLURM_JOB_ID}.csv

# Start GPU monitoring in background (every 5 seconds)
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv -l 5 > "$gpu_stats_file" &
GPU_MONITOR_PID=$!
trap "kill $GPU_MONITOR_PID 2>/dev/null || true" EXIT

# Use scratch venv
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv

# Set HuggingFace cache to scratch
export HF_HOME=$scratch_dir/.cache/huggingface
export TORCH_HOME=$scratch_dir/.cache/torch

# Fix TensorFlow XLA ptxas compatibility issue for Perch/Perch2
# Disable XLA auto-clustering which triggers incompatible ptxas
export TF_XLA_FLAGS="--tf_xla_auto_jit=-1"

# Run benchmark
# Models: BirdNET, Perch, beats
uv run python -m birdclef_adrian.benchmark_inference \
    --models ${MODELS:-beats BirdNET BirdSetConvNeXT BirdSetEfficientNetB1 HawkEars Perch Perch2 RanaSierraeCNN YAMNet} \
    --duration ${DURATION:-60.0} \
    --num-runs ${NUM_RUNS:-10} \
    --warmup-runs ${WARMUP_RUNS:-1} \
    --checkpoint-dir $scratch_dir/checkpoints \
    --output-dir $scratch_dir/outputs/001-inference-benchmark \
    --device cuda

echo ""
echo "Done!"
echo "GPU stats saved to: $gpu_stats_file"
date
