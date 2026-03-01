#!/bin/bash
#SBATCH --job-name=010-indices-soundscape
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH --array=0-19
#SBATCH -t120 -qinferno
#SBATCH --output=logs/indices-soundscape-%A_%a.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
venv=$scratch_dir/.venv-010

cd $repo_dir/user/anthony/experiments/010-teacher-labels

echo "=== Indices soundscape partition ${SLURM_ARRAY_TASK_ID} ==="

$venv/bin/teacher-indices \
    --config configs/teacher.yaml \
    soundscape ${SLURM_ARRAY_TASK_ID}
