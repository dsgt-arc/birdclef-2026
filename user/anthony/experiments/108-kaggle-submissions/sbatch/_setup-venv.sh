#!/bin/bash
# Dedicated venv for exp 108 Kaggle submission tooling.
#
# Shared across submission kernels (ridge, hopfield_v4, hopfield_v1_no_gate).
# CPU-only torch is sufficient for submission inference on Kaggle, but we
# match exp 107's CUDA venv for local dry-runs on the RTX 6000 partition.
#
# Usage: source sbatch/_setup-venv.sh (after optional `module load cuda/12.6.1`)

set -eu

exp108_dir="$HOME/arc/birdclef-2026/user/anthony/experiments/108-kaggle-submissions"
exp107_dir="$HOME/arc/birdclef-2026/user/anthony/experiments/107-hopfield-prototypes"
exp106_dir="$HOME/arc/birdclef-2026/user/anthony/experiments/106-perch-bayesian"

scratch_dir="$HOME/scratch/birdclef-2026"
VENV_DIR="$scratch_dir/.venv-108"
VENV_MARKER="$VENV_DIR/.setup-done-v1"

if [ ! -f "$VENV_MARKER" ]; then
    echo "=== Creating exp 108 venv at $VENV_DIR ==="
    uv venv "$VENV_DIR" --python 3.11
    source "$VENV_DIR/bin/activate"

    # CPU torch is enough for the Kaggle submission surface. Local GPU
    # retrains for exp 107 v4 / v1-no-gate use exp 107's own venv.
    uv pip install --index-url https://download.pytorch.org/whl/cpu torch

    uv pip install \
        numpy \
        pandas \
        polars \
        soundfile \
        onnxruntime \
        huggingface_hub \
        scikit-learn \
        pytest

    touch "$VENV_MARKER"
else
    echo "=== Using existing venv at $VENV_DIR ==="
    source "$VENV_DIR/bin/activate"
fi

# Editable installs pick up live code changes.
echo "Installing editable package from $exp106_dir"
uv pip install -e "$exp106_dir"
echo "Installing editable package from $exp107_dir"
uv pip install -e "$exp107_dir"
echo "Installing editable package from $exp108_dir"
uv pip install -e "$exp108_dir"

cd "$exp108_dir"
