#!/bin/bash
#SBATCH --job-name=009-explore
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -t30 -qinferno
#SBATCH --output=logs/explore-data-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv

cd $repo_dir/user/anthony/experiments/009-birdclef-pipeline

uv run --project $repo_dir/user/anthony birdclef-explore \
    $scratch_dir/data/birdclef-2025/ \
    --output-dir logs/ \
    --output-json logs/data-exploration.json \
    --max-workers 4
