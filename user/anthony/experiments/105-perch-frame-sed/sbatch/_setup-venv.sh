#!/bin/bash
# Shared venv setup for exp 105 sbatch scripts.
#
# Persistent venv on scratch — keeps ONNX tooling cached between jobs.
# No torch / cuda needed for graph inspection; just ONNX + onnxruntime CPU
# + huggingface_hub + onnx-graphsurgeon.
#
# Usage: source sbatch/_setup-venv.sh

set -eu

repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir=$repo_dir/user/anthony/experiments/105-perch-frame-sed

scratch_dir=$(realpath ~/scratch/birdclef-2026)
VENV_DIR="$scratch_dir/.venv-105"
VENV_MARKER="$VENV_DIR/.setup-done"

if [ ! -f "$VENV_MARKER" ]; then
    echo "=== First run: building persistent venv at $VENV_DIR ==="
    uv venv "$VENV_DIR" --python 3.11
    source "$VENV_DIR/bin/activate"

    uv pip install \
        onnx \
        onnxruntime \
        onnx-graphsurgeon \
        huggingface_hub \
        numpy

    touch "$VENV_MARKER"
    echo "=== Persistent venv built ==="
else
    echo "=== Using existing venv at $VENV_DIR ==="
    source "$VENV_DIR/bin/activate"
fi

cd "$exp_dir"
