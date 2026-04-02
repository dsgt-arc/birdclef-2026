#!/usr/bin/env bash
# Push Kaggle kernels for experiment 020.
#
# Usage (from experiment root):
#   kaggle_kernel/push.sh protossm       # Push Perch ProtoSSM v5 kernel
#   kaggle_kernel/push.sh ensemble       # Push Perch + SED ensemble kernel
#
set -euo pipefail

KERNEL_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
    echo "Usage: $0 <mode>"
    echo ""
    echo "Modes:"
    echo "  protossm      Push Perch ProtoSSM v5 kernel (pantanal-distill port)"
    echo "  ensemble      Push Perch + SED ensemble kernel (Phase D)"
    echo ""
    exit 1
}

[[ $# -lt 1 ]] && usage

MODE="$1"
shift

# Resolve kernel metadata
case "$MODE" in
    protossm)   META="kernel-metadata-perch-protossm.json" ;;
    ensemble)   META="kernel-metadata-ensemble.json" ;;
    *)          echo "Unknown mode: $MODE"; usage ;;
esac

cd "$KERNEL_DIR"

if [[ ! -f "$META" ]]; then
    echo "ERROR: $META not found in $KERNEL_DIR"
    exit 1
fi

# Push kernel
echo "=== Pushing kernel: $META ==="
cp "$META" kernel-metadata.json
kaggle kernels push -p .
rm kernel-metadata.json
