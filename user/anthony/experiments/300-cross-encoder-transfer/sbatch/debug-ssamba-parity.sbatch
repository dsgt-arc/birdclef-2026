#!/bin/bash
#SBATCH --job-name=300-ssamba-dbg
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 20 -q inferno
#SBATCH --output=logs/ssamba-dbg-%j.out
set -euo pipefail
mkdir -p logs
export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
source sbatch/_setup-venv.sh
python -u scripts/debug_ssamba_parity.py
