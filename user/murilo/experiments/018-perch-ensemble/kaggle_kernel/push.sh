#!/usr/bin/env bash
# Push a Kaggle kernel using one of the pre-configured metadata files.
#
# Usage (from experiment root):
#   kaggle_kernel/push.sh cache       # Push Notebook A (Perch cache)
#   kaggle_kernel/push.sh scoring     # Push Notebook B (scoring, Perch-only)
#   kaggle_kernel/push.sh ensemble    # Push Notebook B (scoring, SED+Perch ensemble)
#   kaggle_kernel/push.sh perch-raw   # Push raw Perch baseline
#   kaggle_kernel/push.sh perch-only  # Push Perch-only (monolithic, old)
#
set -euo pipefail

KERNEL_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$KERNEL_DIR"

usage() {
    echo "Usage: $0 <mode>"
    echo ""
    echo "Modes:"
    echo "  cache       Notebook A — Perch cache (embeddings + mapped scores)"
    echo "  scoring     Notebook B — Scoring (Perch-only, Phase B)"
    echo "  ensemble    Notebook B — Scoring (SED + Perch, Phase C)"
    echo "  perch-raw   Raw Perch baseline (no probes)"
    echo "  perch-only  Perch-only monolithic kernel (old)"
    exit 1
}

[[ $# -lt 1 ]] && usage

case "$1" in
    cache)      META="kernel-metadata-perch-cache.json" ;;
    scoring)    META="kernel-metadata-scoring.json" ;;
    ensemble)   META="kernel-metadata-scoring-ensemble.json" ;;
    perch-raw)  META="kernel-metadata-perch-raw.json" ;;
    perch-only) META="kernel-metadata-perch-only.json" ;;
    *)          echo "Unknown mode: $1"; usage ;;
esac

if [[ ! -f "$META" ]]; then
    echo "ERROR: $META not found in $SCRIPT_DIR"
    exit 1
fi

echo "Pushing with metadata: $META"
cp "$META" kernel-metadata.json
kaggle kernels push -p .
rm kernel-metadata.json
