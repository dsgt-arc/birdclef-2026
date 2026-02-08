#!/bin/bash
#SBATCH --job-name=006-download
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=4G
#SBATCH -p main
#SBATCH -t60 -qembers
#SBATCH --output=logs/download-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv

cd $repo_dir/user/anthony/experiments/006-ssamba-birdset-hsn

uv run python scripts/download_data.py
