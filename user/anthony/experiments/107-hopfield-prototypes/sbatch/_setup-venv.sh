#!/bin/bash
# Dedicated venv for exp 107 training (CUDA torch).
#
# Exp 106 ships a CPU-only torch (analysis-only workload).  Exp 107 needs
# CUDA torch for Hopfield retrieval training on the RTX 6000.  We keep a
# separate venv at ~/scratch/birdclef-2026/.venv-107 so exp 106's
# analysis runs stay untouched.
#
# Usage: source sbatch/_setup-venv.sh (after `module load cuda/12.6.1`)

set -eu

exp107_dir="$HOME/arc/birdclef-2026/user/anthony/experiments/107-hopfield-prototypes"
exp106_dir="$HOME/arc/birdclef-2026/user/anthony/experiments/106-perch-bayesian"

scratch_dir="$HOME/scratch/birdclef-2026"
VENV_DIR="$scratch_dir/.venv-107"
VENV_MARKER="$VENV_DIR/.setup-done-v2"

if [ ! -f "$VENV_MARKER" ]; then
    echo "=== Creating exp 107 venv at $VENV_DIR (with CUDA torch) ==="
    uv venv "$VENV_DIR" --python 3.11
    source "$VENV_DIR/bin/activate"

    # CUDA torch (matches cuda/12.6.1 module loaded by the sbatch script)
    uv pip install --index-url https://download.pytorch.org/whl/cu126 torch

    # Core deps + dev tools
    uv pip install \
        polars \
        numpy \
        pyyaml \
        scikit-learn \
        tensorboard \
        einops \
        pytest

    # iNat/family fetch script dep
    uv pip install requests

    touch "$VENV_MARKER"
else
    echo "=== Using existing venv at $VENV_DIR ==="
    source "$VENV_DIR/bin/activate"
fi

# Always re-install editable packages to pick up code changes.
echo "Installing editable package from $exp106_dir"
uv pip install -e "$exp106_dir"
echo "Installing editable package from $exp107_dir"
uv pip install -e "$exp107_dir"

cd "$exp107_dir"

# Sanity: verify CUDA torch is actually active
python -c "
import torch
print(f'torch: {torch.__version__}')
print(f'cuda: {torch.version.cuda}')
print(f'cuda_available: {torch.cuda.is_available()}')
"
