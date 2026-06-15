#!/bin/bash
#SBATCH --job-name=300-extract-torch
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=10G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -p gpu-rtx6000
#SBATCH -t 120 -q inferno
#SBATCH --output=logs/extract-torch-%j.out

# Extract one torch-backend encoder's labeled-window embeddings.
#
# Usage:
#   ENCODER=beats     sbatch sbatch/extract-torch.sbatch
#   ENCODER=ast       sbatch sbatch/extract-torch.sbatch
#   ENCODER=eat       sbatch sbatch/extract-torch.sbatch
#   ENCODER=ssamba_hf sbatch sbatch/extract-torch.sbatch
#
# Optional POOLING override (token-pooling sensitivity variant), e.g.:
#   ENCODER=ast VARIANT=frozen-cls POOLING=cls sbatch sbatch/extract-torch.sbatch

set -euo pipefail
mkdir -p logs

# Install THIS worktree's code (else _setup-venv.sh defaults REPO_DIR to ~/arc = the
# exp/301 tree, which lacks new flags like --window-sec). Mirrors extract-focal-torch.sbatch.
export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
echo "REPO_DIR=$REPO_DIR"

module load cuda/12.6.1
source sbatch/_setup-venv.sh

ENCODER=${ENCODER:?set ENCODER=beats|ast|eat|ssamba_hf}
VARIANT=${VARIANT:-frozen}

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 2)" \
    || { echo 'FATAL: CUDA not available' >&2; exit 2; }

birdclef-300 extract \
    --config configs/exp300.yaml \
    --encoder "$ENCODER" \
    --variant "$VARIANT" \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
    ${POOLING:+--pooling "$POOLING"} \
    ${WINDOW_SEC:+--window-sec "$WINDOW_SEC"}
