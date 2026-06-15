#!/bin/bash
#SBATCH --job-name=300-ast-onnx
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 45 -q inferno
#SBATCH --output=logs/ast-onnx-%j.out

# Export AST -> ONNX (+INT8), check eager parity, and benchmark win/s on 4 vCPU
# (== Kaggle's CPU allotment) to see if it clears the 90-min wall. CPU-only job.
#   REPO_DIR=$PWD/../../../.. sbatch sbatch/export-ast-onnx.sbatch
set -euo pipefail
mkdir -p logs artifacts

# Pin REPO_DIR to THIS worktree (else _setup-venv.sh runs ~/arc = the exp/301 tree).
export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
echo "REPO_DIR=$REPO_DIR"

source sbatch/_setup-venv.sh
# ONNX export + dynamic quant + ORT benchmark (not in the base venv closure).
uv pip install onnx onnxruntime >/dev/null 2>&1

python scripts/export_ast_onnx.py "$@"
