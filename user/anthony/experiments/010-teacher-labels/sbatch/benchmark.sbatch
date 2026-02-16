#!/bin/bash
#SBATCH --job-name=010-benchmark
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH -t60 -qinferno
#SBATCH --output=logs/benchmark-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
venv=$scratch_dir/.venv-010

cd $repo_dir/user/anthony/experiments/010-teacher-labels

echo "=== Teacher model benchmark ==="

$venv/bin/teacher-benchmark \
    --config configs/teacher.yaml \
    --output $scratch_dir/outputs/exp010/benchmark.json
