#!/bin/bash
#SBATCH --job-name=300-ssamba-mambaonnx
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 45 -q inferno
#SBATCH --output=logs/ssamba-mambaonnx-%j.out

# Export SSAMBA at native 10 s geometry with the mamba-onnx fused custom op,
# verify ONNX==eager parity, and benchmark CPU throughput on 4 vCPU (== Kaggle).
# 4 cpus-per-task so the benchmark previews the real Kaggle wall.
#   REPO_DIR=$PWD/../../../.. sbatch sbatch/export-ssamba-mambaonnx.sbatch
set -euo pipefail
mkdir -p logs
export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
source sbatch/_setup-venv.sh        # activates venv, cd's into the exp dir

WHEEL="$REPO_DIR/vendor/mamba-onnx-wheels/mamba_onnx-0.1.0-py3-none-manylinux_2_17_x86_64.whl"
echo "ensuring onnxruntime + local mamba-onnx wheel ($WHEEL)"
# onnxruntime already in .venv-300; wheel is a local file (no internet on compute nodes).
uv pip install onnxruntime "$WHEEL"
python -c "import mamba_onnx; print('backends:', mamba_onnx.available_backends())"

BENCH_THREADS=4 python -u scripts/export_ssamba_mambaonnx.py

echo "=== result json ==="
cat artifacts/ssamba_onnx/export_bench.json
