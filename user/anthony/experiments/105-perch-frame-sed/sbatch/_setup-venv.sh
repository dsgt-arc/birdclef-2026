#!/bin/bash
# Shared venv setup for exp 105 sbatch scripts.
#
# Persistent venv on scratch — keeps torch + onnxruntime + lightning cached
# between jobs. The editable install of the exp 105 package is re-run every
# time so code changes are picked up.
#
# Marker bumped to -v2 when stage 1/2 deps (torch, lightning, etc.) were added
# on top of the original inspection-only venv. Deleting the old marker forces
# a rebuild, which is fine — the inspection job doesn't need to re-run.
#
# Usage: source sbatch/_setup-venv.sh

set -eu

repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir=$repo_dir/user/anthony/experiments/105-perch-frame-sed

scratch_dir=$(realpath ~/scratch/birdclef-2026)
VENV_DIR="$scratch_dir/.venv-105"
VENV_MARKER="$VENV_DIR/.setup-done-v2"

if [ ! -f "$VENV_MARKER" ]; then
    echo "=== Building persistent venv at $VENV_DIR (v2: +torch +lightning) ==="
    rm -rf "$VENV_DIR"
    uv venv "$VENV_DIR" --python 3.11
    source "$VENV_DIR/bin/activate"

    # Base ONNX + inspection tools
    uv pip install \
        onnx \
        onnxruntime \
        onnx-graphsurgeon \
        huggingface_hub \
        numpy \
        pyyaml

    # Torch + torchaudio (CPU wheels — we run Perch on CPU, and heads are tiny).
    # The matching CUDA wheels are not needed for phase 1.
    uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

    # Stage 2 — training stack
    uv pip install \
        pytorch-lightning \
        torchmetrics \
        pandas \
        pyarrow \
        soundfile

    touch "$VENV_MARKER"
    echo "=== Persistent venv built ==="
else
    echo "=== Using existing venv at $VENV_DIR ==="
    source "$VENV_DIR/bin/activate"
fi

# Always re-install editable exp 105 package to pick up code changes.
uv pip install -e "$exp_dir" --quiet

cd "$exp_dir"
