#!/bin/bash
#SBATCH --job-name=007-precompute
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t240 -qinferno
#SBATCH --output=logs/precompute-events-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv

cd $repo_dir/user/anthony/experiments/007-pretrained-ssamba-transfer

uv run --project $repo_dir/user/anthony python precompute_mels.py \
    --config config-default-16x16.yaml \
    --cache-dir $scratch_dir/cache/007-hsn-events
