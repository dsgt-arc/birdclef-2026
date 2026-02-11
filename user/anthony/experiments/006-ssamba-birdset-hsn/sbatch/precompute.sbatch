#!/bin/bash
#SBATCH --job-name=006-precompute
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t120 -qinferno
#SBATCH --output=logs/precompute-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv

cd $repo_dir/user/anthony/experiments/006-ssamba-birdset-hsn

uv run python precompute_mels.py --config config.yaml --cache-dir $scratch_dir/cache/006-hsn
