#!/usr/bin/env bash
# Build datasets and push Kaggle kernels for experiment 018.
#
# Usage (from experiment root):
#   kaggle_kernel/push.sh single           # Build dataset + push single-kernel (recommended)
#   kaggle_kernel/push.sh single --skip-dataset  # Push kernel only (dataset already uploaded)
#   kaggle_kernel/push.sh cache            # Push Notebook A (Perch cache)
#   kaggle_kernel/push.sh perch-raw        # Push raw Perch baseline
#
set -euo pipefail

KERNEL_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENT_DIR="$(dirname "$KERNEL_DIR")"
SKIP_DATASET=false

# Dataset paths
LABELED_EMB_DATASET_DIR="/storage/project/ps-clef2026_birdclef-0/shared/datasets/birdclef-2026/perch-labeled-embeddings"

usage() {
    echo "Usage: $0 <mode> [--skip-dataset]"
    echo ""
    echo "Modes:"
    echo "  single      Build labeled embeddings dataset + push single-kernel scoring"
    echo "  cache       Push Notebook A — Perch cache (embeddings + mapped scores)"
    echo "  scoring     Push Notebook B — Scoring (two-notebook chain, deprecated)"
    echo "  ensemble    Push Notebook B — Scoring (SED + Perch, Phase C)"
    echo "  perch-raw   Push raw Perch baseline (no probes)"
    echo "  perch-only  Push Perch-only monolithic kernel (old)"
    echo ""
    echo "Options:"
    echo "  --skip-dataset  Skip dataset build/upload (for kernel-only pushes)"
    exit 1
}

[[ $# -lt 1 ]] && usage

MODE="$1"
shift

# Parse optional flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-dataset) SKIP_DATASET=true; shift ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# Resolve kernel metadata
case "$MODE" in
    single)     META="kernel-metadata-single.json" ;;
    cache)      META="kernel-metadata-perch-cache.json" ;;
    scoring)    META="kernel-metadata-scoring.json" ;;
    ensemble)   META="kernel-metadata-scoring-ensemble.json" ;;
    perch-raw)  META="kernel-metadata-perch-raw.json" ;;
    perch-only) META="kernel-metadata-perch-only.json" ;;
    *)          echo "Unknown mode: $MODE"; usage ;;
esac

cd "$KERNEL_DIR"

if [[ ! -f "$META" ]]; then
    echo "ERROR: $META not found in $KERNEL_DIR"
    exit 1
fi

# --- Step 1: Build + upload dataset (for 'single' mode) ---
if [[ "$MODE" == "single" && "$SKIP_DATASET" == "false" ]]; then
    echo "=== Step 1: Build labeled embeddings dataset ==="
    cd "$EXPERIMENT_DIR"
    python scripts/build_labeled_embeddings_dataset.py

    echo ""
    echo "=== Step 2: Upload dataset to Kaggle ==="
    cd "$LABELED_EMB_DATASET_DIR"
    # Try create first, then version if it already exists
    if ! kaggle datasets create -p . -r zip 2>/dev/null; then
        echo "  Dataset exists, creating new version..."
        kaggle datasets version -p . -m "Update labeled embeddings" -r zip
    fi
    echo ""
fi

# --- Step 2 (or only step): Push kernel ---
echo "=== Pushing kernel: $META ==="
cd "$KERNEL_DIR"
cp "$META" kernel-metadata.json
kaggle kernels push -p .
rm kernel-metadata.json
