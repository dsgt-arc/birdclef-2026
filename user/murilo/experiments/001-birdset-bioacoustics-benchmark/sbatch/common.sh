#!/bin/bash
# Common functions and variables for BirdSet benchmark SLURM jobs.
#
# Usage: source this file from job scripts
#   source "$(dirname "$0")/../common.sh"

# Exit on error, undefined variables
set -eu

# ============================================================================
# Directory Configuration (derived from script location)
# ============================================================================

# This file is at: experiments/001-birdset-bioacoustics-benchmark/sbatch/common.sh
# Derive paths relative to this script
_COMMON_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${_COMMON_SH_DIR}/.." && pwd)"
USER_DIR="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
REPO_DIR="$(cd "${USER_DIR}/../.." && pwd)"

# Scratch directory
SCRATCH_DIR="${HOME}/scratch/birdclef-2026"

# Artifacts output directory
ARTIFACTS_DIR="${EXPERIMENT_DIR}/artifacts"

# ============================================================================
# Environment Setup
# ============================================================================

setup_environment() {
    # UV venv location in scratch (XDG_CACHE_HOME handles HF/torch caches via ~/.bashrc)
    export UV_PROJECT_ENVIRONMENT="${SCRATCH_DIR}/.venv"

    # Suppress TensorFlow warnings
    export TF_CPP_MIN_LOG_LEVEL=3
    export TF_ENABLE_ONEDNN_OPTS=0
    export PYTHONWARNINGS=ignore

    # Change to repo directory
    cd "${REPO_DIR}"

    # Sync dependencies and install user package
    uv sync
    uv pip install -e "${EXPERIMENT_DIR}"
}

# ============================================================================
# GPU Monitoring
# ============================================================================

start_gpu_monitoring() {
    local log_dir="${EXPERIMENT_DIR}/logs"
    local log_file="${log_dir}/gpu_stats-${SLURM_JOB_ID:-local}.csv"

    # Ensure log directory exists
    mkdir -p "${log_dir}"

    # Start GPU monitoring in background (every 5 seconds)
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv -l 5 > "$log_file" &
    GPU_MONITOR_PID=$!

    # Cleanup on exit
    trap "kill $GPU_MONITOR_PID 2>/dev/null || true" EXIT

    echo "GPU monitoring started (PID: $GPU_MONITOR_PID, log: $log_file)"
}

# ============================================================================
# Job Info
# ============================================================================

print_job_info() {
    local job_name="${1:-BirdSet Benchmark}"
    echo "=============================================="
    echo "Job: ${job_name}"
    echo "=============================================="
    echo "Job ID: ${SLURM_JOB_ID:-local}"
    echo "Node: $(hostname)"
    echo "Date: $(date)"
    echo "Repo: ${REPO_DIR}"
    echo "Experiment: ${EXPERIMENT_DIR}"
    echo ""
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
        echo ""
    fi
}

# ============================================================================
# Artifact Paths
# ============================================================================

get_inference_output_dir() {
    local model="$1"
    local dataset="$2"
    echo "${ARTIFACTS_DIR}/inference/${model}/${dataset}"
}

check_inference_complete() {
    local model="$1"
    local dataset="$2"
    local output_dir
    output_dir=$(get_inference_output_dir "$model" "$dataset")

    if [[ -f "${output_dir}/metrics.json" ]]; then
        return 0  # Complete
    else
        return 1  # Not complete
    fi
}

# ============================================================================
# BirdSet Configuration
# ============================================================================

# Available models for Phase 2A (Direct Inference)
BIRDSET_MODELS=("birdset_convnext" "birdset_efficientnet_b1")

# Available datasets
BIRDSET_DATASETS=("HSN" "PER" "NES" "UHH" "POW" "SSW" "SNE" "NBP")
