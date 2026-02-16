#!/bin/bash
#SBATCH --job-name=010-bambird-cluster
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH -t60 -qinferno
#SBATCH --output=logs/bambird-cluster-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
venv=$scratch_dir/.venv-010

cd $repo_dir/user/anthony/experiments/010-teacher-labels

echo "=== BamBird cluster (all species) ==="

$venv/bin/teacher-bambird cluster \
    --config configs/teacher.yaml
