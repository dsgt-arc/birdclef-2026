#!/bin/bash
#SBATCH --job-name=300-ast-int8-val
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 45 -q inferno
#SBATCH --output=logs/ast-int8-val-%j.out

# Downstream gate for INT8: head + INT8-ONNX vs head + eager on the 739 soundscapes.
# If macro AUROC holds, INT8 ships with the existing head despite the 0.993 cosine.
#   REPO_DIR=$PWD/../../../.. sbatch sbatch/validate-ast-int8.sbatch
set -euo pipefail
mkdir -p logs artifacts
export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
echo "REPO_DIR=$REPO_DIR"

source sbatch/_setup-venv.sh
uv pip install onnxruntime scikit-learn >/dev/null 2>&1

python scripts/validate_ast_int8.py "$@"
