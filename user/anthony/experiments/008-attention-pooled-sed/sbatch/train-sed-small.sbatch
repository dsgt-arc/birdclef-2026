#!/bin/bash
#SBATCH --job-name=008-sed-128x1-small
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-rtx6000
#SBATCH -t1440 -qinferno
#SBATCH --output=logs/train-sed-small-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv

cd $repo_dir/user/anthony/experiments/008-attention-pooled-sed

module load cuda/12.1.1

ckpt=$scratch_dir/checkpoints/ssamba/ssamba_small_400.pth

CKPT_PATH=${CKPT_PATH:-}

uv run --project $repo_dir/user/anthony python train.py \
    --config config-sed-128x1-small.yaml \
    --pretrained-ckpt $ckpt \
    --run-name sed-128x1-small \
    --chunk-cache-dir $scratch_dir/cache/008-chunk-index \
    ${CKPT_PATH:+--ckpt-path $CKPT_PATH}
