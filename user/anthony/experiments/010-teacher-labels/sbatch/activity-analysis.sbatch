#!/bin/bash
#SBATCH --job-name=010-activity
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=32G
#SBATCH -t60 -qinferno
#SBATCH --output=logs/activity-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
venv=$scratch_dir/.venv-010

cd $repo_dir/user/anthony/experiments/010-teacher-labels

echo "=== Activity analysis ==="

$venv/bin/python scripts/activity_analysis.py \
    $scratch_dir/outputs/exp010/activity.json
