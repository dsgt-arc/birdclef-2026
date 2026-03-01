#!/bin/bash
#SBATCH --job-name=007-framelevel-128x1
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-rtx6000
#SBATCH -t1440 -qinferno
#SBATCH --output=logs/train-framelevel-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv

cd $repo_dir/user/anthony/experiments/007-pretrained-ssamba-transfer

module load cuda/12.1.1

cache_dir=$scratch_dir/cache/007-hsn-events
ckpt=$scratch_dir/checkpoints/ssamba/ssamba_tiny_400.pth

CKPT_PATH=${CKPT_PATH:-}

uv run --project $repo_dir/user/anthony python train.py \
    --config config-framelevel-128x1.yaml \
    --cache-dir $cache_dir \
    --pretrained-ckpt $ckpt \
    --run-name framelevel-128x1 \
    ${CKPT_PATH:+--ckpt-path $CKPT_PATH}
