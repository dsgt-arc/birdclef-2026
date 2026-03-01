#!/bin/bash
#SBATCH --job-name=bench-ssast
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -t240 -qembers
#SBATCH --output=logs/benchmark-ssast-%j.out
set -eu

echo "Running SSAST encoder benchmark on ESC-50"
echo "Job ID: $SLURM_JOB_ID"
hostname
date
nvidia-smi

scratch_dir=~/scratch/birdclef-2026
repo_dir=~/arc/birdclef-2026
gpu_stats_file=logs/gpu_stats-ssast-${SLURM_JOB_ID}.csv

# Check if checkpoint exists
ckpt_path=$scratch_dir/checkpoints/ssast/SSAST-Base-Patch-400.pth
if [ ! -f "$ckpt_path" ]; then
    echo "ERROR: SSAST checkpoint not found at $ckpt_path"
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

cd $repo_dir

# Install timm 0.4.5 for SSAST compatibility
uv sync
uv pip install -e user/anthony
uv pip install timm==0.4.5

# Run SSAST benchmark
uv run python -m birdclef.benchmark \
    --encoder ssast \
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
