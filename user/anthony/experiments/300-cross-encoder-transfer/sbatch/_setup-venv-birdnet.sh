#!/bin/bash
# Separate TensorFlow venv for the BirdNET anchor (bioacoustics-model-zoo serves
# BirdNET's TFLite model). Kept apart from the torch venv so TF and torch never
# share a process. Mirrors exp 010's .venv-010 setup.
#
# Usage: source sbatch/_setup-venv-birdnet.sh

set -eu

repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir=$repo_dir/user/anthony/experiments/300-cross-encoder-transfer

scratch_dir=$(realpath ~/scratch/birdclef-2026)
VENV_DIR="$scratch_dir/.venv-300-birdnet"
VENV_MARKER="$VENV_DIR/.setup-done"

if [ ! -f "$VENV_MARKER" ]; then
    echo "=== First run: building BirdNET venv at $VENV_DIR ==="
    uv venv "$VENV_DIR" --python 3.11 --seed
    source "$VENV_DIR/bin/activate"

    uv pip install bioacoustics-model-zoo
    uv pip install polars soundfile pyyaml numpy pandas

    touch "$VENV_MARKER"
    echo "=== BirdNET venv built ==="
else
    echo "=== Using existing venv at $VENV_DIR ==="
    source "$VENV_DIR/bin/activate"
fi

uv pip install -e "$exp_dir"

cd "$exp_dir"
