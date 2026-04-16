#!/bin/bash
#SBATCH --job-name=exp107-test-cache-sidecar
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail

exp107_dir="$HOME/arc/birdclef-2026/user/anthony/experiments/107-hopfield-prototypes"
cd "$exp107_dir"
mkdir -p logs

source sbatch/_setup-venv.sh

export PYTHONUNBUFFERED=1
ROOT_DIR="${TMPDIR:-/tmp}/exp107-cache-smoke-${SLURM_JOB_ID:-$$}"
rm -rf "$ROOT_DIR"
mkdir -p "$ROOT_DIR"

echo "=== Synthetic sidecar cache tests ==="
echo "=== root=$ROOT_DIR ==="

python -u scripts/make_synthetic_cache_fixture.py --root "$ROOT_DIR"
pytest tests/test_cache.py tests/test_perch_features.py -q
python -u scripts/smoke_cache_sidecar.py --root "$ROOT_DIR"

echo "=== Sidecar cache smoke test passed ==="
