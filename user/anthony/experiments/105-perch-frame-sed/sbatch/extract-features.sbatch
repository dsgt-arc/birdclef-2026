#!/bin/bash
#SBATCH --job-name=105-extract
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 120
#SBATCH -q inferno
#SBATCH --output=logs/extract-%j.out

# Stage 1: run Perch v2 ONNX over labeled soundscape + nocall windows and
# write a pooled-embedding cache. CPU-only (Perch is light and CUDA provider
# isn't in the venv).
#
# Usage:
#   cd user/anthony/experiments/105-perch-frame-sed
#   sbatch sbatch/extract-features.sbatch                 # full cache
#   sbatch sbatch/extract-features.sbatch --limit 10      # pilot (10 files)
#
# Any additional args are forwarded to `birdclef-105 extract`.

set -eu
mkdir -p logs artifacts

source sbatch/_setup-venv.sh

echo ""
echo "=== Stage 1: extract Perch pooled embeddings ==="
birdclef-105 extract --config configs/exp105.yaml "$@"

echo ""
echo "=== Done ==="
