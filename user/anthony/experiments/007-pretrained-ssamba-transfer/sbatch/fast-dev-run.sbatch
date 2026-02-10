#!/bin/bash
#SBATCH --job-name=007-fast-dev
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-rtx6000
#SBATCH -t60 -qembers
#SBATCH --output=logs/fast-dev-run-%j.out

set -eu

# Parametric: pass CONFIG env var, e.g.:
#   CONFIG=config-default-16x16.yaml sbatch sbatch/fast-dev-run.sbatch
#   CONFIG=config-framelevel-128x1.yaml sbatch sbatch/fast-dev-run.sbatch
CONFIG=${CONFIG:-config-default-16x16.yaml}

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv

cd $repo_dir/user/anthony/experiments/007-pretrained-ssamba-transfer

module load cuda/12.1.1

cache_dir=$scratch_dir/cache/007-hsn-events
ckpt=$scratch_dir/checkpoints/ssamba/ssamba_tiny_400.pth

uv run --project $repo_dir/user/anthony python train.py \
    --config $CONFIG \
    --fast-dev-run \
    --num-workers 2 \
    --cache-dir $cache_dir \
    --pretrained-ckpt $ckpt
