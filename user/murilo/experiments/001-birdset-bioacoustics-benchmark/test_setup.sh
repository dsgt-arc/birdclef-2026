#!/bin/bash
#
# Interactive test script for environment setup and validation
# Run this in an interactive GPU session to set up and verify installation
#
# This script:
#   1. Checks GPU availability
#   2. Sets up environment variables
#   3. Installs uv package manager
#   4. Creates virtual environment
#   5. Installs dependencies from pyproject.toml
#   6. Installs PyTorch with CUDA support
#   7. Verifies PyTorch CUDA support
#   8. Runs comprehensive pytest test suite
#
# Usage:
#   salloc -p gpu-rtx6000 --gres=gpu:1 -c 4 --mem=32G -t 2:00:00
#   bash test_setup.sh
#

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "=========================================="
echo "BirdSet Benchmark - Setup & Validation"
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
export EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../../../.." && pwd)"

# Use TMPDIR if available (SLURM job), otherwise use ~/tmp for interactive session
if [ -n "${TMPDIR:-}" ]; then
    VENV_BASE="$TMPDIR"
    echo "  Using SLURM TMPDIR for fast local I/O"
else
    VENV_BASE="$HOME/tmp"
    mkdir -p "$VENV_BASE"
    echo "  Using ~/tmp for interactive session"
fi

JOB_ID="${SLURM_JOB_ID:-$$}"
export JOB_VENV_DIR="$VENV_BASE/birdset-benchmark-$JOB_ID"
export JOB_VENV="$JOB_VENV_DIR/.venv"

echo "  - Experiment dir: $EXPERIMENT_DIR"
echo "  - Project root: $PROJECT_ROOT"
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
cd "$PROJECT_ROOT/user/murilo"
uv venv "$JOB_VENV" --python 3.11
print_status "Virtual environment created"
echo ""

# Step 5: Activate environment
echo "Step 5: Activating virtual environment..."
source "$JOB_VENV/bin/activate"
print_status "Environment activated"
echo ""

# Step 6: Install dependencies
echo "Step 6: Installing dependencies from pyproject.toml..."
echo "  This may take 5-10 minutes for first install..."
uv pip install -e "$PROJECT_ROOT/user/murilo"
print_status "Dependencies installed"
echo ""

# Step 6b: Install PyTorch with CUDA support
echo "Step 6b: Installing PyTorch with CUDA 12.1 support..."
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
print_status "PyTorch with CUDA installed"
echo ""

# Step 7: Verify PyTorch and CUDA
echo "Step 7: Verifying PyTorch CUDA support..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
print_status "PyTorch CUDA verified"
echo ""

# Step 8: Run pytest test suite
echo "Step 8: Running pytest test suite..."
cd "$PROJECT_ROOT/user/murilo"
pytest tests/001-birdset-bioacoustics-benchmark/ -v --tb=short
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo "  Check output above for details"
    exit 1
fi
echo ""

# Step 9: Summary
echo "=========================================="
echo -e "${GREEN}✓ ENVIRONMENT SETUP AND VALIDATION COMPLETE${NC}"
echo "=========================================="
echo ""
echo "Environment is ready for experiments!"
echo ""
echo "Test Results:"
echo "  - GPU: Available ($(nvidia-smi --query-gpu=name --format=csv,noheader))"
echo "  - PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  - CUDA: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else \"N/A\")')"
echo "  - All pytest tests: Passed"
echo ""
echo "Next steps:"
echo "  1. Implement data pipeline (Phase 2)"
echo "  2. Create embedding extraction script (Phase 4)"
echo "  3. Train linear probes (Phase 5)"
echo ""
echo "Python location: $(which python)"
echo "Virtual environment: $JOB_VENV"
echo ""
echo "To use this environment in your scripts:"
echo "  source $JOB_VENV/bin/activate"
echo ""
echo "⚠️  Note: This environment is temporary and will be cleaned up."
echo "   SLURM jobs automatically create isolated venvs in \$TMPDIR."
echo ""
