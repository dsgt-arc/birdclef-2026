#!/bin/bash
# Download dependency wheels for offline Kaggle kernel installation.
#
# Downloads manylinux wheels compatible with Kaggle's Python 3.12 + x86_64 environment.
# These get uploaded as a separate Kaggle dataset (birdclef-deps) and installed
# offline in the submission kernel via pip --no-index --find-links.
#
# Usage:
#   source .venv/bin/activate  # needs pip available
#   bash scripts/build_deps.sh

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
DEPS_DIR="$EXPERIMENT_DIR/artifacts/dist/dependencies"

echo "=== Building Kaggle dependency wheels ==="
echo "Output: $DEPS_DIR"

mkdir -p "$DEPS_DIR"

# Download standard PyPI wheels for Kaggle's environment
# Use multiple --platform flags to cover both manylinux_2_17 (older) and _2_28 (newer)
echo ""
echo "--- Downloading PyPI wheels ---"
pip download \
    onnxruntime \
    --platform manylinux_2_28_x86_64 \
    --platform manylinux_2_17_x86_64 \
    --platform linux_x86_64 \
    --python-version 3.12 \
    --only-binary=:all: \
    -d "$DEPS_DIR"

# Create requirements.txt for reference
cat > "$DEPS_DIR/requirements.txt" <<'EOF'
onnxruntime
EOF

echo ""
echo "=== Dependencies staged ==="
echo "Files:"
ls -lh "$DEPS_DIR"
echo ""
echo "Total size: $(du -sh "$DEPS_DIR" | cut -f1)"
echo ""
echo "Next step: upload with"
echo "  python scripts/upload_kaggle.py --model protoclr --push-kernel"
