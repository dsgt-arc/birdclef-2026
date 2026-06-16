#!/bin/bash
#SBATCH --job-name=300-ssamba-bench
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=12G
#SBATCH -p cpu-small
#SBATCH -t 40 -q inferno
#SBATCH --output=logs/ssamba-bench-%j.out

# Decide the SSAMBA CPU forward path (eager pure-PyTorch vs unrolled ONNX) at 4 vCPU ==
# Kaggle. UNBUFFERED so partial results survive. The unrolled ONNX timed out under ORT;
# this isolates whether eager is fast enough and whether DISABLE_ALL rescues the ONNX.
#   REPO_DIR=$PWD/../../../.. sbatch sbatch/bench-ssamba-cpu.sbatch
set -euo pipefail
mkdir -p logs
export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
echo "REPO_DIR=$REPO_DIR"
source sbatch/_setup-venv.sh
uv pip install onnxruntime >/dev/null 2>&1

python -u scripts/bench_ssamba_cpu.py "$@"
