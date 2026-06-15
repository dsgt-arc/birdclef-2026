#!/bin/bash
#SBATCH --job-name=300-eat-onnx
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 45 -q inferno
#SBATCH --output=logs/eat-onnx-%j.out

# Export EAT (cls, finetune_AS2M) -> ONNX, check eager parity, benchmark win/s on 4 vCPU
# (== Kaggle's CPU allotment). Tracing extract_features sidesteps trust_remote_code for
# the offline kernel. CPU-only; the finetuned checkpoint is HF-cached from extraction.
#   REPO_DIR=$PWD/../../../.. sbatch sbatch/export-eat-onnx.sbatch
set -euo pipefail
mkdir -p logs artifacts

# Pin REPO_DIR to THIS worktree (else _setup-venv.sh runs ~/arc = the exp/301 tree).
export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
echo "REPO_DIR=$REPO_DIR"

source sbatch/_setup-venv.sh
# ONNX export + ORT benchmark (not in the base venv closure).
uv pip install onnx onnxruntime >/dev/null 2>&1

python scripts/export_eat_onnx.py "$@"
