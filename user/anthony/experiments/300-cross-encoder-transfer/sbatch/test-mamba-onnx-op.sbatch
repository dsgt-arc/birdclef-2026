#!/bin/bash
#SBATCH --job-name=300-test-mamba-op
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 20 -q inferno
#SBATCH --output=logs/test-mamba-op-%j.out
set -euo pipefail; mkdir -p logs
export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
source sbatch/_setup-venv.sh
uv pip install pytest >/dev/null 2>&1 || true
cd "$REPO_DIR/vendor/mamba-onnx"
# -o addopts="" drops the repo's --benchmark-skip; -k zig skips the cpp backend (no libomp)
python -m pytest tests/test_ops.py -o addopts="" -k "zig" -q 2>&1 \
  | grep -vE "shape_type_inference|UpdateReliable" | tail -45
