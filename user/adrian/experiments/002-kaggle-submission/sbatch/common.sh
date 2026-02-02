#!/bin/bash
# Common functions for Kaggle submission sbatch scripts
#
# Usage: source this file from job scripts
#   source ~/arc/birdclef-2026/user/adrian/experiments/002-kaggle-submission/sbatch/common.sh

set -eu

# ============================================================================
# Directory Configuration (derived from $HOME)
# ============================================================================

# Derive all paths from $HOME since SLURM copies scripts to temp directories
REPO_DIR="${HOME}/arc/birdclef-2026"
USER_DIR="${REPO_DIR}/user/adrian"
EXPERIMENT_DIR="${USER_DIR}/experiments/002-kaggle-submission"
SBATCH_DIR="${EXPERIMENT_DIR}/sbatch"

# Scratch directory
SCRATCH_DIR="${HOME}/scratch/birdclef-2026"

# Artifacts output directory
ARTIFACTS_DIR="${EXPERIMENT_DIR}/artifacts"
DIST_DIR="${ARTIFACTS_DIR}/dist"

# ============================================================================
# Environment Setup
# ============================================================================

setup_environment() {
    # UV venv location in scratch
    export UV_PROJECT_ENVIRONMENT="${SCRATCH_DIR}/.venv"
    
    # HuggingFace/Torch cache locations
    export HF_HOME="${SCRATCH_DIR}/.cache/huggingface"
    export TORCH_HOME="${SCRATCH_DIR}/.cache/torch"
    
    # Suppress warnings
    export TF_CPP_MIN_LOG_LEVEL=3
    export PYTHONWARNINGS=ignore
    
    cd "${REPO_DIR}"
}

# ============================================================================
# Job Info
# ============================================================================

print_job_info() {
    local job_name="${1:-Kaggle Submission Build}"
    echo "=============================================="
    echo "Job: ${job_name}"
    echo "=============================================="
    echo "Job ID: ${SLURM_JOB_ID:-local}"
    echo "Node: $(hostname)"
    echo "Date: $(date)"
    echo "Repo: ${REPO_DIR}"
    echo "Experiment: ${EXPERIMENT_DIR}"
    echo "Dist Dir: ${DIST_DIR}"
    echo ""
}

# ============================================================================
# Artifact Helpers
# ============================================================================

ensure_dist_dirs() {
    mkdir -p "${DIST_DIR}/dependencies"
    mkdir -p "${DIST_DIR}/code"
    mkdir -p "${DIST_DIR}/weights"
}

clean_dist_dir() {
    local subdir="$1"
    echo "Cleaning ${DIST_DIR}/${subdir}..."
    rm -rf "${DIST_DIR}/${subdir:?}"/*
}

# ============================================================================
# Network Check
# ============================================================================

check_network() {
    echo "Testing outbound network connectivity..."
    if curl -s --connect-timeout 5 https://pypi.org > /dev/null 2>&1; then
        echo "✓ Network connectivity to PyPI: OK"
        return 0
    else
        echo "✗ Network connectivity to PyPI: FAILED"
        echo "  Run dependency builds on login node instead."
        return 1
    fi
}
