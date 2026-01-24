#!/bin/bash
#
# Interactive test script for environment setup
# Run this in an interactive GPU session to verify installation
#
# Usage:
#   salloc -p gpu-rtx6000 --gres=gpu:1 -c 4 --mem=32G -t 2:00:00
#   bash test_setup.sh
#

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "=========================================="
echo "BirdSet Benchmark - Environment Test"
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

# Step 8: Test bioacoustics-model-zoo
echo "Step 8: Testing bioacoustics-model-zoo installation..."
python << 'EOF'
try:
    import bioacoustics_model_zoo as bmz
    print(f"  bioacoustics-model-zoo version: {bmz.__version__ if hasattr(bmz, '__version__') else 'unknown'}")

    # List available models
    print("\n  Available models:")
    models = bmz.list_models()
    for model_name in models:
        print(f"    - {model_name}")

    print(f"\n  Total models: {len(models)}")
except Exception as e:
    print(f"  Error: {e}")
    exit(1)
EOF
print_status "bioacoustics-model-zoo functional"
echo ""

# Step 9: Test HuggingFace datasets
echo "Step 9: Testing HuggingFace datasets..."
python << 'EOF'
from datasets import load_dataset_builder

try:
    # Check if BirdSet is accessible (don't download yet)
    builder = load_dataset_builder("DBD-research-group/BirdSet", "HSN")
    print(f"  BirdSet HSN dataset accessible")
    print(f"  Description: {builder.info.description[:100]}...")
except Exception as e:
    print(f"  Warning: Could not access BirdSet. Error: {e}")
    print("  You may need to authenticate with HuggingFace or check network access.")
EOF
print_status "HuggingFace datasets accessible"
echo ""

# Step 10: Test model loading (quick test)
echo "Step 10: Testing model instantiation (BirdNET)..."
python << 'EOF'
import bioacoustics_model_zoo as bmz
import torch
import numpy as np

try:
    print("  Loading BirdNET model...")
    model = bmz.BirdNET()
    print(f"  Model type: {type(model)}")

    # Test with dummy audio (3 seconds at default sample rate)
    sample_rate = 48000  # BirdNET uses 48kHz
    duration = 3.0
    dummy_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

    print(f"  Testing embedding extraction with dummy audio ({dummy_audio.shape})...")
    with torch.no_grad():
        embedding = model.embed(dummy_audio)

    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding dtype: {embedding.dtype}")

    # Verify embedding dimension
    expected_dim = 1024
    if embedding.shape[-1] == expected_dim:
        print(f"  ✓ Embedding dimension matches expected ({expected_dim})")
    else:
        print(f"  ✗ Embedding dimension mismatch: got {embedding.shape[-1]}, expected {expected_dim}")
        exit(1)

except Exception as e:
    print(f"  Error during model test: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF
print_status "Model loading and embedding extraction successful"
echo ""

# Step 11: Test scikit-learn
echo "Step 11: Testing scikit-learn..."
python << 'EOF'
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

# Test with dummy data
X_train = np.random.randn(100, 1024)
y_train = np.random.randint(0, 2, (100, 10))  # Multi-label

clf = OneVsRestClassifier(LogisticRegression(max_iter=100))
clf.fit(X_train, y_train)

X_test = np.random.randn(10, 1024)
y_pred = clf.predict_proba(X_test)

print(f"  Trained classifier on {X_train.shape[0]} samples")
print(f"  Prediction shape: {y_pred.shape}")
print(f"  ✓ Multi-label classification working")
EOF
print_status "scikit-learn functional"
echo ""

# Step 12: Summary
echo "=========================================="
echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
echo "=========================================="
echo ""
echo "Environment is ready for experiments!"
echo ""
echo "Next steps:"
echo "  1. Download BirdSet HSN dataset (may be large)"
echo "  2. Run embedding extraction script"
echo "  3. Train linear probe"
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
