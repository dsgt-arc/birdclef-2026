#!/bin/bash
# Download dependency wheels for offline Kaggle kernel installation.
#
# Downloads manylinux wheels compatible with Kaggle's Python 3.12 + x86_64 environment.
# Uploaded as Kaggle dataset (birdclef-2026-deps) and installed offline in submission kernel.
#
# Usage:
#   bash scripts/build_deps.sh

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
DEPS_DIR="$EXPERIMENT_DIR/artifacts/deps"

echo "=== Building Kaggle dependency wheels ==="
echo "Output: $DEPS_DIR"

rm -rf "$DEPS_DIR"
mkdir -p "$DEPS_DIR"

# Download wheels for Kaggle's Python 3.12 x86_64 environment
# onnxruntime: ONNX inference
# librosa + soundfile: audio loading and mel spectrogram computation
# numpy/pandas are preinstalled on Kaggle, but download anyway for version safety
echo ""
echo "--- Downloading PyPI wheels ---"
pip download \
    onnxruntime \
    librosa \
    soundfile \
    numpy \
    pandas \
    --platform manylinux_2_28_x86_64 \
    --platform manylinux_2_17_x86_64 \
    --platform linux_x86_64 \
    --python-version 3.12 \
    --only-binary=:all: \
    -d "$DEPS_DIR"

echo ""
echo "=== Dependencies staged ==="
echo "Files:"
ls -lh "$DEPS_DIR"
echo ""
echo "Total size: $(du -sh "$DEPS_DIR" | cut -f1)"
