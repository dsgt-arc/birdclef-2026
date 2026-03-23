#!/bin/bash
#SBATCH --job-name=010-birdnet-focal-2026
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --array=0-49
#SBATCH -t240 -qinferno
#SBATCH --output=logs/birdnet-focal-2026-%A_%a.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
venv=$scratch_dir/.venv-010

cd $repo_dir/user/anthony/experiments/010-teacher-labels

echo "=== BirdNET focal 2026 partition ${SLURM_ARRAY_TASK_ID} ==="

$venv/bin/teacher-birdnet \
    --config configs/teacher-2026.yaml \
    focal ${SLURM_ARRAY_TASK_ID}
