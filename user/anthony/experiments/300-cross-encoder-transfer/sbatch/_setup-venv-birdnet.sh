#!/bin/bash
# Separate TensorFlow venv for the BirdNET anchor (bioacoustics-model-zoo serves
# BirdNET's TFLite model). Kept apart from the torch venv so TF and torch never
# share a process. Mirrors exp 010's .venv-010 setup.
#
# Usage: source sbatch/_setup-venv-birdnet.sh

set -eu

# REPO_DIR override (matches _setup-venv.sh) lets a job run from a specific worktree
# (e.g. exp/300) instead of whatever ~/arc/birdclef-2026 points at — required so the
# focal-extraction code (extract_birdnet_focal) is picked up, not stale ~/arc code.
repo_dir=$(realpath "${REPO_DIR:-$HOME/arc/birdclef-2026}")
exp_dir=$repo_dir/user/anthony/experiments/300-cross-encoder-transfer

scratch_dir=$(realpath ~/scratch/birdclef-2026)
VENV_DIR="$scratch_dir/.venv-300-birdnet"
VENV_MARKER="$VENV_DIR/.setup-done"

if [ ! -f "$VENV_MARKER" ]; then
    echo "=== First run: building BirdNET venv at $VENV_DIR ==="
    uv venv "$VENV_DIR" --python 3.11 --seed
    source "$VENV_DIR/bin/activate"

    uv pip install bioacoustics-model-zoo
    uv pip install ai-edge-litert            # tflite runtime BirdNET loads at model init
    uv pip install polars soundfile pyyaml numpy pandas

    touch "$VENV_MARKER"
    echo "=== BirdNET venv built ==="
else
    echo "=== Using existing venv at $VENV_DIR ==="
    source "$VENV_DIR/bin/activate"
fi

uv pip install -e "$exp_dir"

cd "$exp_dir"
