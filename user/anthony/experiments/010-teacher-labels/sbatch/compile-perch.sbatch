#!/bin/bash
#SBATCH --job-name=010-compile-perch
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=16G
#SBATCH -t120 -qinferno
#SBATCH --output=logs/compile-perch-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
venv=$scratch_dir/.venv-010

cd $repo_dir/user/anthony/experiments/010-teacher-labels

echo "=== Compile Perch to TFLite ==="
echo "Venv: $venv"
echo "Output: $scratch_dir/outputs/exp010/perch.tflite"

$venv/bin/teacher-compile-perch \
    --output-path $scratch_dir/outputs/exp010/perch.tflite
