#!/bin/bash
#SBATCH --job-name=bench-ssamba-hf-cpu
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=16 --mem-per-cpu=4G
#SBATCH -t720 -qinferno
#SBATCH --output=logs/benchmark-ssamba-hf-cpu-%j.out
set -eu

# Model size: tiny, small, or base (default: small)
MODEL_SIZE=${MODEL_SIZE:-small}

echo "Running SSAMBA-HF-$MODEL_SIZE encoder benchmark on ESC-50 (CPU only)"
echo "This uses the HuggingFace port (no CUDA kernels required)"
echo "Job ID: $SLURM_JOB_ID"
hostname
date

scratch_dir=~/scratch/birdclef-2026
repo_dir=~/arc/birdclef-2026

# Check if checkpoint exists
ckpt_path=$scratch_dir/checkpoints/ssamba/ssamba_${MODEL_SIZE}_400.pth
if [ ! -f "$ckpt_path" ]; then
    echo "ERROR: SSAMBA checkpoint not found at $ckpt_path"
    echo "Please run download-part-c-checkpoints.sbatch first"
    exit 1
fi

export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export HF_HOME=$scratch_dir/.cache/huggingface

cd $repo_dir/user/anthony

# Install dependencies
uv sync

# Run SSAMBA-HF benchmark on CPU
uv run python -m birdclef.benchmark \
    --encoder ssamba-hf-$MODEL_SIZE \
    --esc50-root $repo_dir/vendor/ESC-50 \
    --checkpoint-dir $scratch_dir/checkpoints \
    --output-dir $scratch_dir/outputs/esc50-benchmark-partc \
    --device cpu \
    --batch-size 8 \
    --epochs 100 \
    --lr 1e-3

echo ""
echo "Done!"
date
