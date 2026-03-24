#!/bin/bash
#SBATCH --job-name=102-spot-check
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=2 --mem-per-cpu=4G
#SBATCH -t10 -qinferno
#SBATCH --output=logs/spot-check-%j.out

set -eu

repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir=$repo_dir/user/anthony/experiments/102-efficientnet-sed

cd $exp_dir

venv=~/scratch/birdclef-2026/.venv-010
source $venv/bin/activate

python scripts/spot_check_activity.py \
    --activity artifacts/focal_activity.parquet
