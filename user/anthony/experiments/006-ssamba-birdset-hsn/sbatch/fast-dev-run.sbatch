#!/bin/bash
#SBATCH --job-name=006-fast-dev
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

cd $repo_dir/user/anthony/experiments/006-ssamba-birdset-hsn

module load cuda/12.1.1

cache_dir=$scratch_dir/cache/006-hsn

uv run python train.py --config config.yaml --fast-dev-run --num-workers 2 --cache-dir $cache_dir
