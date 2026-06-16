#!/bin/bash
#SBATCH --job-name=300-eager-thr
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 15 -q inferno
#SBATCH --output=logs/eager-thr-%j.out
set -euo pipefail; mkdir -p logs
export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
source sbatch/_setup-venv.sh
export OMP_NUM_THREADS=4
THREADS=4 python -u scripts/bench_ssamba_eager_threads.py
