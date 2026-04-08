#!/usr/bin/env bash
# Push Kaggle kernel for experiment 023.
#
# Usage (from anywhere):
#   user/murilo/experiments/023-perch-sed-ensemble/kaggle_kernel/push.sh
#
set -euo pipefail

KERNEL_DIR="$(cd "$(dirname "$0")" && pwd)"
META="kernel-metadata-ensemble.json"

cd "$KERNEL_DIR"

if [[ ! -f "$META" ]]; then
    echo "ERROR: $META not found in $KERNEL_DIR"
    exit 1
fi

echo "=== Pushing kernel: $META ==="
cp "$META" kernel-metadata.json
kaggle kernels push -p .
rm kernel-metadata.json
