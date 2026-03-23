#!/bin/bash
#SBATCH --job-name=010-birdnet-soundscape-2026
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --array=0-19
#SBATCH -t120 -qinferno
#SBATCH --output=logs/birdnet-soundscape-2026-%A_%a.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
venv=$scratch_dir/.venv-010

cd $repo_dir/user/anthony/experiments/010-teacher-labels

echo "=== BirdNET soundscape 2026 partition ${SLURM_ARRAY_TASK_ID} ==="

$venv/bin/teacher-birdnet \
    --config configs/teacher-2026.yaml \
    soundscape ${SLURM_ARRAY_TASK_ID}
