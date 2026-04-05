#!/bin/bash
#SBATCH --job-name=105-train
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 30
#SBATCH -q inferno
#SBATCH --output=logs/train-%j.out

# Stage 2: train a head on cached Perch pooled embeddings.
# CPU-only — the head is a single Linear layer (or tiny MLP), training is
# sub-minute per epoch on CPU. GPU would be wasteful for ~1k samples.
#
# Usage:
#   cd user/anthony/experiments/105-perch-frame-sed
#   sbatch sbatch/train-head.sbatch --head linear
#   sbatch sbatch/train-head.sbatch --head mlp --run-name mlp-v1
#
# All args after the script name are forwarded to `birdclef-105 train`.

set -eu
mkdir -p logs artifacts

source sbatch/_setup-venv.sh

echo ""
echo "=== Stage 2: train head ==="
birdclef-105 train --config configs/exp105.yaml "$@"

echo ""
echo "=== Done ==="
