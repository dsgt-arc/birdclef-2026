#!/bin/bash
#SBATCH --job-name=100-eda
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=4G
#SBATCH -t60 -qinferno

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir=$repo_dir/user/anthony/experiments/100-eda
data_dir=/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026

cd $exp_dir
mkdir -p logs artifacts

# redirect stdout/stderr to log file (avoids SLURM relative path issues)
exec > logs/eda-${SLURM_JOB_ID}.out 2>&1

# Set up venv in TMPDIR (fast local NVMe)
JOB_VENV_DIR="$TMPDIR/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"
uv pip install -e "$exp_dir"

echo "=== Running EDA pipeline ==="
echo "Data: $data_dir"
echo "Artifacts: $exp_dir/artifacts"
echo "Job: $SLURM_JOB_ID"

eda-pipeline \
    --data-dir "$data_dir" \
    --artifacts-dir "$exp_dir/artifacts" \
    --workers 1 \
    --n-audio-workers 8

echo "=== Done ==="
ls -lh $exp_dir/artifacts/
