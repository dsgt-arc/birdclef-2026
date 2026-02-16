#!/bin/bash
#SBATCH --job-name=010-stats
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=32G
#SBATCH -t120 -qinferno
#SBATCH --output=logs/stats-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
venv=$scratch_dir/.venv-010

cd $repo_dir/user/anthony/experiments/010-teacher-labels

echo "=== Teacher label statistics ==="

$venv/bin/teacher-stats \
    --config configs/teacher.yaml \
    --output $scratch_dir/outputs/exp010/stats.json

echo "=== Generating plots ==="
$venv/bin/python scripts/plot_stats.py \
    $scratch_dir/outputs/exp010/stats.json \
    $scratch_dir/outputs/exp010/teacher_stats.png

echo "=== Checking output completeness ==="
$venv/bin/python scripts/check_outputs.py configs/teacher.yaml
