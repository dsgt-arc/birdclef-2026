#!/bin/bash
#SBATCH --job-name=010-tests
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH -t60 -qinferno
#SBATCH --output=logs/tests-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
venv=$scratch_dir/.venv-010

cd $repo_dir/user/anthony/experiments/010-teacher-labels

echo "=== Running teacher label tests ==="
echo "Venv: $venv"
echo "Python: $($venv/bin/python --version)"

# Run all tests with verbose output
# - test_utils: always runs (no data needed)
# - test_bambird/birdnet/perch: skipped if data not found
$venv/bin/python -m pytest tests/ -v 2>&1
