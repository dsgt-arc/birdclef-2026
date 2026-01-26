#!/bin/bash
#SBATCH --job-name=bench-ssamba-hf
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -t360 -qembers
#SBATCH --output=logs/benchmark-ssamba-hf-%j.out
set -eu

# Model size: tiny, small, or base (default: small)
MODEL_SIZE=${MODEL_SIZE:-small}

echo "Running SSAMBA-HF-$MODEL_SIZE encoder benchmark on ESC-50"
echo "This uses the HuggingFace port (no CUDA kernels required)"
echo "Job ID: $SLURM_JOB_ID"
hostname
date
nvidia-smi

scratch_dir=~/scratch/birdclef-2026
repo_dir=~/arc/birdclef-2026
gpu_stats_file=logs/gpu_stats-ssamba-hf-${MODEL_SIZE}-${SLURM_JOB_ID}.csv

# Check if checkpoint exists
ckpt_path=$scratch_dir/checkpoints/ssamba/ssamba_${MODEL_SIZE}_400.pth
if [ ! -f "$ckpt_path" ]; then
    echo "ERROR: SSAMBA checkpoint not found at $ckpt_path"
    echo "Please run download-part-c-checkpoints.sbatch first"
    exit 1
fi

# Start GPU monitoring
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv -l 5 > "$gpu_stats_file" &
GPU_MONITOR_PID=$!
trap "kill $GPU_MONITOR_PID 2>/dev/null || true" EXIT

export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export HF_HOME=$scratch_dir/.cache/huggingface

cd $repo_dir/user/anthony

# Install dependencies (no mamba-ssm or causal-conv1d required!)
uv sync

# Run SSAMBA-HF benchmark
uv run python -m birdclef.benchmark \
    --encoder ssamba-hf-$MODEL_SIZE \
    --esc50-root $repo_dir/vendor/ESC-50 \
    --checkpoint-dir $scratch_dir/checkpoints \
    --output-dir $scratch_dir/outputs/esc50-benchmark-partc \
    --batch-size 16 \
    --epochs 100 \
    --lr 1e-3

echo ""
echo "Done!"
echo "GPU stats saved to: $gpu_stats_file"
date
