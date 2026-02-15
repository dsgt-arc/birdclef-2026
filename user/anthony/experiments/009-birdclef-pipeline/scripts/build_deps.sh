#!/bin/bash
# Download dependency wheels for offline Kaggle kernel installation.
#
# Downloads manylinux wheels compatible with Kaggle's Python 3.12 + x86_64 environment.
# Also downloads mamba-onnx pre-built wheel from GitHub releases.
#
# Usage:
#   bash scripts/build_deps.sh

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
DEPS_DIR="$EXPERIMENT_DIR/artifacts/dist/dependencies"

echo "=== Building Kaggle dependency wheels ==="
echo "Output: $DEPS_DIR"

mkdir -p "$DEPS_DIR"

# Download standard PyPI wheels
# Use multiple --platform flags to cover both manylinux_2_17 (older) and _2_28 (newer) wheels
echo ""
echo "--- Downloading PyPI wheels ---"
pip download \
    onnxruntime \
    numpy \
    pandas \
    soundfile \
    librosa \
    --platform manylinux_2_28_x86_64 \
    --platform manylinux_2_17_x86_64 \
    --platform linux_x86_64 \
    --python-version 3.12 \
    --only-binary=:all: \
    -d "$DEPS_DIR"

# Download mamba-onnx pre-built wheel
echo ""
echo "--- Downloading mamba-onnx wheel ---"
MAMBA_ONNX_URL="https://github.com/acmiyaguchi/mamba-onnx/releases/latest/download/mamba_onnx-0.1.0-py3-none-manylinux_2_17_x86_64.whl"
wget -nv -P "$DEPS_DIR" "$MAMBA_ONNX_URL" || {
    echo "WARNING: Failed to download mamba-onnx wheel from GitHub releases"
    echo "You may need to build it manually or update the URL"
}

# Create requirements.txt for reference
cat > "$DEPS_DIR/requirements.txt" <<'EOF'
onnxruntime
mamba_onnx
librosa
soundfile
numpy
pandas
EOF

echo ""
echo "=== Dependencies staged ==="
echo "Files:"
ls -lh "$DEPS_DIR"
echo ""
echo "Total size: $(du -sh "$DEPS_DIR" | cut -f1)"
