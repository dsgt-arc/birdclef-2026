#!/bin/bash
#
# Create and set up virtual environment for BirdSet benchmark experiments
# Run this in an interactive GPU session to set up the environment
#
# This script:
#   1. Checks GPU availability
#   2. Sets up environment variables
#   3. Installs uv package manager
#   4. Creates virtual environment
#   5. Installs dependencies from the experiment's pyproject.toml
#   6. Installs PyTorch with CUDA support
#   7. Verifies PyTorch CUDA support
#
# Usage:
#   salloc -p gpu-rtx6000 --gres=gpu:1 -c 4 --mem=32G -t 2:00:00
#   bash create_venv.sh --exp-num 4
#
# Options:
#   --exp-num NUM  Experiment number (e.g., 1, 2, 3, 4) - required
#   -h, --help     Show this help message
#

set -e  # Exit on error
set -u  # Exit on undefined variable

# ============================================================================
# Argument parsing
# ============================================================================

EXP_NUM=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: bash create_venv.sh --exp-num NUM"
    echo ""
    echo "Options:"
    echo "  --exp-num NUM  Experiment number (e.g., 1, 2, 3, 4)"
    echo "  -h, --help     Show this help message"
    echo ""
    echo "Available experiments:"
    for d in "$SCRIPT_DIR"/experiments/[0-9][0-9][0-9]-*/; do
        [ -d "$d" ] && echo "                   $(basename "$d")"
    done
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp-num)
            if [[ -z "${2:-}" ]]; then
                echo "ERROR: --exp-num requires a value"
                usage 1
            fi
            EXP_NUM="$2"
            shift 2
            ;;
        -h|--help)
            usage 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            usage 1
            ;;
    esac
done

if [[ -z "$EXP_NUM" ]]; then
    echo "ERROR: --exp-num is required"
    usage 1
fi

# Map experiment number to directory name dynamically
# Zero-pad to 3 digits (e.g., 4 -> 004) and glob for matching directory
EXP_NUM_PADDED=$(printf "%03d" "$EXP_NUM")
EXP_DIR_MATCH=("$SCRIPT_DIR"/experiments/${EXP_NUM_PADDED}-*/)

if [[ ! -d "${EXP_DIR_MATCH[0]}" ]]; then
    echo "ERROR: No experiment directory found matching ${EXP_NUM_PADDED}-*"
    echo ""
    echo "Available experiments:"
    for d in "$SCRIPT_DIR"/experiments/[0-9][0-9][0-9]-*/; do
        [ -d "$d" ] && echo "  $(basename "$d")"
    done
    exit 1
fi

if [[ ${#EXP_DIR_MATCH[@]} -gt 1 ]]; then
    echo "ERROR: Multiple directories match ${EXP_NUM_PADDED}-*:"
    for d in "${EXP_DIR_MATCH[@]}"; do
        echo "  $(basename "$d")"
    done
    exit 1
fi

EXP_DIR_NAME="$(basename "${EXP_DIR_MATCH[0]}")"

echo "=========================================="
echo "BirdSet Benchmark - Environment Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1${NC}"
    else
        echo -e "${RED}✗ $1 FAILED${NC}"
        exit 1
    fi
}

# Step 1: Check we're on a GPU node
echo "Step 1: Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    print_status "GPU detected"
else
    echo -e "${RED}✗ No GPU detected. Please run in interactive GPU session.${NC}"
    exit 1
fi
echo ""

# Step 2: Set up environment variables
echo "Step 2: Setting up environment variables..."
export PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXPERIMENT_DIR="$SCRIPT_DIR/experiments/$EXP_DIR_NAME"

if [[ ! -d "$EXPERIMENT_DIR" ]]; then
    echo -e "${RED}✗ Experiment directory not found: $EXPERIMENT_DIR${NC}"
    exit 1
fi

# Use TMPDIR if available (SLURM job), otherwise use /tmp for interactive session
# IMPORTANT: Never use ~/tmp or home dir — 20GB quota on PACE
if [ -n "${TMPDIR:-}" ]; then
    VENV_BASE="$TMPDIR"
    echo "  Using SLURM TMPDIR for fast local I/O"
else
    VENV_BASE="/tmp"
    echo "  Using /tmp for interactive session"
fi

JOB_ID="${SLURM_JOB_ID:-$$}"
export JOB_VENV_DIR="$VENV_BASE/birdset-benchmark-$JOB_ID"
export JOB_VENV="$JOB_VENV_DIR/.venv"

echo "  - Script dir: $SCRIPT_DIR"
echo "  - Project root: $PROJECT_ROOT"
echo "  - Experiment: $EXP_DIR_NAME"
echo "  - Experiment dir: $EXPERIMENT_DIR"
echo "  - Job ID: $JOB_ID"
echo "  - Virtual env: $JOB_VENV"
print_status "Environment variables set"
echo ""

# Step 3: Load uv module
echo "Step 3: Loading uv for package management..."
if ! command -v uv &> /dev/null; then
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi
uv --version
print_status "uv available"
echo ""

# Step 4: Create virtual environment
echo "Step 4: Setting up venv at $JOB_VENV..."
mkdir -p "$JOB_VENV_DIR"
cd "$EXPERIMENT_DIR"
uv venv "$JOB_VENV" --python 3.11
print_status "Virtual environment created"
echo ""

# Step 5: Activate environment
echo "Step 5: Activating virtual environment..."
source "$JOB_VENV/bin/activate"
print_status "Environment activated"
echo ""

# Step 6: Install dependencies
echo "Step 6: Installing dependencies from $EXP_DIR_NAME/pyproject.toml..."
echo "  This may take 5-10 minutes for first install..."
uv pip install -e "$EXPERIMENT_DIR[dev]"
print_status "Dependencies installed"
echo ""

# Step 6b: Install PyTorch with CUDA support
echo "Step 6b: Installing PyTorch with CUDA 12.1 support..."
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
print_status "PyTorch with CUDA installed"
echo ""

# Step 7: Verify PyTorch and CUDA
echo "Step 7: Verifying PyTorch CUDA support..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else None}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
print_status "PyTorch CUDA verified"
echo ""

# Summary
echo "=========================================="
echo -e "${GREEN}✓ ENVIRONMENT SETUP COMPLETE${NC}"
echo "=========================================="
echo ""
echo "Environment is ready for experiment: $EXP_DIR_NAME"
echo ""
echo "Summary:"
echo "  - Experiment: $EXP_DIR_NAME"
echo "  - GPU: Available ($(nvidia-smi --query-gpu=name --format=csv,noheader))"
echo "  - PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  - CUDA: $(python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'N/A')")"
echo ""
echo "Python location: $(which python)"
echo "Virtual environment: $JOB_VENV"
echo ""
echo "To use this environment in your scripts:"
echo "  source $JOB_VENV/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest tests/001-birdset-bioacoustics-benchmark/ -v --tb=short"
echo ""
echo -e "${YELLOW}⚠️  Note: This environment is temporary and will be cleaned up.${NC}"
echo "   SLURM jobs automatically create isolated venvs in \$TMPDIR."
echo ""
