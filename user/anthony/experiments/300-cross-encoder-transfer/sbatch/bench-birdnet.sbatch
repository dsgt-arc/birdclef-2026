#!/bin/bash
#SBATCH --job-name=300-bench-birdnet
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 30 -q inferno
#SBATCH --output=logs/bench-birdnet-%j.out
# BirdNET benchmark in the TF venv, same 4-vCPU control.
set -euo pipefail; mkdir -p logs
export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
source sbatch/_setup-venv-birdnet.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
BENCH_CLIPS=8 python -u scripts/bench_all_encoders.py birdnet
