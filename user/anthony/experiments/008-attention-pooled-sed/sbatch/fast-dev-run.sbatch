#!/bin/bash
#SBATCH --job-name=008-fast-dev
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-rtx6000
#SBATCH -t60 -qembers
#SBATCH --output=logs/fast-dev-run-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv

cd $repo_dir/user/anthony/experiments/008-attention-pooled-sed

module load cuda/12.1.1

ckpt=$scratch_dir/checkpoints/ssamba/ssamba_tiny_400.pth

uv run --project $repo_dir/user/anthony python train.py \
    --config config-sed-128x1.yaml \
    --fast-dev-run \
    --num-workers 2 \
    --chunk-cache-dir $scratch_dir/cache/008-chunk-index \
    --pretrained-ckpt $ckpt
