#!/bin/bash
#SBATCH --job-name=010-setup-test
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH -t120 -qinferno
#SBATCH --output=logs/setup-and-test-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
venv=$scratch_dir/.venv-010
exp_dir=$repo_dir/user/anthony/experiments/010-teacher-labels

cd $exp_dir

echo "=== Step 1: Create venv and install package ==="
export UV_PROJECT_ENVIRONMENT=$venv
# Pre-build broken sdists that uv can't handle
uv venv $venv --python 3.11 --seed
$venv/bin/pip install grad-cam==1.5.5
uv sync --extra compile --extra dev

echo ""
echo "=== Step 2: Compile Perch to TFLite ==="
$venv/bin/teacher-compile-perch \
    --output-path $scratch_dir/outputs/exp010/perch.tflite

echo ""
echo "=== Step 3: Run tests ==="
$venv/bin/python -m pytest tests/ -v 2>&1
