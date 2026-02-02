#!/bin/bash
#SBATCH --job-name=kaggle-build-weights
#SBATCH --account=paceship-dsgt_clef2025
#SBATCH -N1 -n1 --cpus-per-task=2 --mem-per-cpu=2G
#SBATCH -t15 -qembers
#SBATCH --output=logs/build-weights-%j.out
set -eu

# Package model weights for Kaggle submission

# Source common functions (absolute path since SLURM copies scripts)
source ~/arc/birdclef-2026/user/adrian/experiments/002-kaggle-submission/sbatch/common.sh
print_job_info "Build Weights Bundle"

ensure_dist_dirs
clean_dist_dir "weights"

WEIGHTS_DIR="${DIST_DIR}/weights"
CHECKPOINT_SRC="${EXPERIMENT_DIR}/checkpoints"

echo "Packaging model weights..."

if [[ -d "${CHECKPOINT_SRC}" ]]; then
    # Copy all checkpoint files
    echo "  Source: ${CHECKPOINT_SRC}"
    cp -v "${CHECKPOINT_SRC}"/*.pt "${WEIGHTS_DIR}/" 2>/dev/null || true
    cp -v "${CHECKPOINT_SRC}"/*.safetensors "${WEIGHTS_DIR}/" 2>/dev/null || true
    cp -v "${CHECKPOINT_SRC}"/*.pth "${WEIGHTS_DIR}/" 2>/dev/null || true
    
    # Copy config files if present
    cp -v "${CHECKPOINT_SRC}"/*.yaml "${WEIGHTS_DIR}/" 2>/dev/null || true
    cp -v "${CHECKPOINT_SRC}"/*.json "${WEIGHTS_DIR}/" 2>/dev/null || true
else
    echo "WARNING: Checkpoint directory not found: ${CHECKPOINT_SRC}"
    echo "Creating empty weights directory..."
fi

echo ""
echo "Weights saved to: ${WEIGHTS_DIR}"
ls -la "${WEIGHTS_DIR}" 2>/dev/null || echo "(empty)"

echo ""
echo "Done!"
date
