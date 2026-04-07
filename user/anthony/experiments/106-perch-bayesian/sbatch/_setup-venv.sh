#!/bin/bash
# Shared venv setup for exp 106 sbatch scripts.
# Reuses the exp 105 venv on scratch (same deps: onnxruntime, torch, soundfile).
# Only re-installs the exp 106 editable package.
#
# Usage: source sbatch/_setup-venv.sh

set -eu

repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir=$repo_dir/user/anthony/experiments/106-perch-bayesian

scratch_dir=$(realpath ~/scratch/birdclef-2026)
VENV_DIR="$scratch_dir/.venv-106"
VENV_MARKER="$VENV_DIR/.setup-done-v1"

if [ ! -f "$VENV_MARKER" ]; then
    echo "=== Building persistent venv at $VENV_DIR ==="
    rm -rf "$VENV_DIR"
    uv venv "$VENV_DIR" --python 3.11
    source "$VENV_DIR/bin/activate"

    uv pip install \
        onnxruntime \
        huggingface_hub \
        numpy \
        pyyaml \
        polars \
        soundfile

    # CPU-only torch for Perch ONNX + torchaudio resampling
    uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

    touch "$VENV_MARKER"
    echo "=== Persistent venv built ==="
else
    echo "=== Using existing venv at $VENV_DIR ==="
    source "$VENV_DIR/bin/activate"
fi

# Always re-install editable package to pick up code changes.
uv pip install -e "$exp_dir" --quiet

cd "$exp_dir"
