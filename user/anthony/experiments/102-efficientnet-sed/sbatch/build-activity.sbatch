#!/bin/bash
#SBATCH --job-name=102-activity
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -t30 -qinferno
#SBATCH --output=logs/build-activity-%j.out

set -eu

repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir=$repo_dir/user/anthony/experiments/102-efficientnet-sed

cd $exp_dir

venv=~/scratch/birdclef-2026/.venv-010
source $venv/bin/activate

python scripts/build_activity_scores.py \
    --output artifacts/focal_activity.parquet
