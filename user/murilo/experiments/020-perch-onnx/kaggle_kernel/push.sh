#!/usr/bin/env bash
# Push Kaggle kernels for experiment 020.
#
# Usage (from experiment root):
#   kaggle_kernel/push.sh protossm           # Push monolith kernel (V17, backward compat)
#   kaggle_kernel/push.sh protossm-train     # Push V18 training kernel (saves artifacts)
#   kaggle_kernel/push.sh protossm-infer     # Push V18 inference kernel (loads artifacts)
#   kaggle_kernel/push.sh ensemble           # Push Perch + SED ensemble kernel
#
set -euo pipefail

KERNEL_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
    echo "Usage: $0 <mode>"
    echo ""
    echo "Modes:"
    echo "  protossm          Push monolith kernel (train+infer, V17)"
    echo "  protossm-train    Push V18 training kernel (artifacts only)"
    echo "  protossm-infer    Push V18 inference kernel (uses pretrained artifacts)"
    echo "  ensemble          Push Perch + SED ensemble kernel (Phase D)"
    echo ""
    exit 1
}

[[ $# -lt 1 ]] && usage

MODE="$1"
shift

# Resolve kernel metadata and optional env override
ENV_OVERRIDE=""
case "$MODE" in
    protossm)       META="kernel-metadata-perch-protossm.json" ;;
    protossm-train) META="kernel-metadata-train.json"; ENV_OVERRIDE="PIPELINE_MODE=train_only" ;;
    protossm-infer) META="kernel-metadata-infer.json" ;;
    ensemble)       META="kernel-metadata-ensemble.json" ;;
    *)              echo "Unknown mode: $MODE"; usage ;;
esac

cd "$KERNEL_DIR"

if [[ ! -f "$META" ]]; then
    echo "ERROR: $META not found in $KERNEL_DIR"
    exit 1
fi

# For train mode, prepend PIPELINE_MODE override to the script
CODE_FILE=$(python3 -c "import json; print(json.load(open('$META'))['code_file'])")
BACKUP=""
if [[ -n "$ENV_OVERRIDE" ]]; then
    echo "Setting $ENV_OVERRIDE in $CODE_FILE"
    BACKUP="${CODE_FILE}.bak"
    cp "$CODE_FILE" "$BACKUP"
    # Insert env override after the "# ruff: noqa" line
    sed -i "2i import os; os.environ['PIPELINE_MODE'] = 'train_only'  # AUTO-INJECTED by push.sh" "$CODE_FILE"
fi

# Push kernel
echo "=== Pushing kernel: $META ==="
cp "$META" kernel-metadata.json
kaggle kernels push -p .
rm kernel-metadata.json

# Restore original script if we modified it
if [[ -n "$BACKUP" ]]; then
    mv "$BACKUP" "$CODE_FILE"
    echo "Restored original $CODE_FILE"
fi
