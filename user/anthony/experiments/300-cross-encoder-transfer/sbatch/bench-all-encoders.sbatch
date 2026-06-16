#!/bin/bash
#SBATCH --job-name=300-bench-all
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 60 -q inferno
#SBATCH --output=logs/bench-all-%j.out
# Controlled 4-vCPU (Kaggle-equivalent) inference benchmark, torch/ONNX encoders.
set -euo pipefail; mkdir -p logs
export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
source sbatch/_setup-venv.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for enc in perch beats ast eat ssamba; do
  echo "===== $enc ====="
  BENCH_CLIPS=8 python -u scripts/bench_all_encoders.py "$enc" || echo "[$enc] FAILED"
done
echo "===== summary ====="; cat artifacts/bench_all.json
