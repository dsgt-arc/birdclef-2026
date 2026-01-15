#!/bin/bash
#SBATCH --job-name=bench-ssamba
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -t360 -qembers
#SBATCH --output=logs/benchmark-ssamba-%j.out
set -eu

# Model size: tiny, small, or base (default: base)
MODEL_SIZE=${MODEL_SIZE:-base}

echo "Running SSAMBA-$MODEL_SIZE encoder benchmark on ESC-50"
echo "Job ID: $SLURM_JOB_ID"
hostname
date
nvidia-smi

scratch_dir=~/scratch/birdclef-2026
repo_dir=~/arc/birdclef-2026
gpu_stats_file=logs/gpu_stats-ssamba-${MODEL_SIZE}-${SLURM_JOB_ID}.csv

# Check if checkpoint exists
ckpt_path=$scratch_dir/checkpoints/ssamba/ssamba_${MODEL_SIZE}_400.pth
if [ ! -f "$ckpt_path" ]; then
    echo "ERROR: SSAMBA checkpoint not found at $ckpt_path"
    echo "Please run download-part-c-checkpoints.sbatch first"
    exit 1
fi

# Check if Vim repo exists
vim_dir=$repo_dir/vendor/ssamba/Vim
if [ ! -d "$vim_dir" ]; then
    echo "ERROR: Vim repository not found at $vim_dir"
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

# Install dependencies
uv sync
uv pip install -e user/anthony

# Install mamba-ssm and causal-conv1d (requires CUDA)
# Note: This may take a while to compile
uv pip install causal-conv1d>=1.1.0
uv pip install mamba-ssm>=1.0.0

# Install Vim mamba from the vendored repo
cd $vim_dir/vim/mamba-1p1p1
uv pip install -e .
cd $repo_dir

# Run SSAMBA benchmark
uv run python -m birdclef.benchmark \
    --encoder ssamba-$MODEL_SIZE \
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
