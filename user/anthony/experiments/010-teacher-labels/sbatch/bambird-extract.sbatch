#!/bin/bash
#SBATCH --job-name=010-bambird-extract
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH --array=0-49
#SBATCH -t180 -qinferno
#SBATCH --output=logs/bambird-extract-%A_%a.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
venv=$scratch_dir/.venv-010

cd $repo_dir/user/anthony/experiments/010-teacher-labels

echo "=== BamBird extract-features partition ${SLURM_ARRAY_TASK_ID} ==="

PYTHONUNBUFFERED=1 $venv/bin/teacher-bambird extract-features \
    --config configs/teacher.yaml \
    ${SLURM_ARRAY_TASK_ID}
