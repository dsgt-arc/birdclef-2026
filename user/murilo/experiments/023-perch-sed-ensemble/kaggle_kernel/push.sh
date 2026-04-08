#!/usr/bin/env bash
# Push Kaggle kernel with config override for experiment 023.
# Each config creates a SEPARATE Kaggle notebook (unique kernel ID).
#
# Usage:
#   kaggle_kernel/push.sh <config_name>
#   kaggle_kernel/push.sh ensemble_w03
#   kaggle_kernel/push.sh ensemble_w07
#
# Available configs: ls configs/*.yaml
#
set -euo pipefail

KERNEL_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$KERNEL_DIR/../configs"
BASE_META="kernel-metadata-ensemble.json"
CODE_FILE="submission_ensemble.py"
KAGGLE_USER="murilogustineli"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config_name>"
    echo ""
    echo "Available configs:"
    ls "$CONFIG_DIR"/*.yaml 2>/dev/null | xargs -I{} basename {} .yaml | sed 's/^/  /'
    exit 1
fi

CONFIG_NAME="$1"
CONFIG_FILE="$CONFIG_DIR/${CONFIG_NAME}.yaml"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config not found: $CONFIG_FILE"
    echo ""
    echo "Available configs:"
    ls "$CONFIG_DIR"/*.yaml 2>/dev/null | xargs -I{} basename {} .yaml | sed 's/^/  /'
    exit 1
fi

cd "$KERNEL_DIR"

# Convert YAML config to Python dict literal (JSON)
OVERRIDE=$(python3 -c "
import yaml, json
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
cfg.pop('name', None)
cfg.pop('description', None)
print(json.dumps(cfg))
")

# Generate unique kernel metadata from config name
# e.g., ensemble_w03 -> birdclef-2026-perch-sed-ensemble-w03
KERNEL_SUFFIX="${CONFIG_NAME#ensemble_}"  # strip 'ensemble_' prefix
KERNEL_ID="${KAGGLE_USER}/birdclef-2026-perch-sed-${KERNEL_SUFFIX}"
KERNEL_TITLE="BirdCLEF 2026 Perch+SED ${KERNEL_SUFFIX}"

python3 -c "
import json
with open('$BASE_META') as f:
    meta = json.load(f)
meta['id'] = '$KERNEL_ID'
meta['title'] = '$KERNEL_TITLE'
with open('kernel-metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
print(f'Kernel ID: {meta[\"id\"]}')
print(f'Title: {meta[\"title\"]}')
"

echo "=== Pushing kernel: $CONFIG_NAME ==="
echo "Override: $OVERRIDE"

# Inject override into kernel
cp "$CODE_FILE" "${CODE_FILE}.bak"
sed -i "s|^CFG_OVERRIDE = {}.*|CFG_OVERRIDE = $OVERRIDE  # from configs/${CONFIG_NAME}.yaml|" "$CODE_FILE"

# Push
kaggle kernels push -p .
rm kernel-metadata.json

# Restore original script
mv "${CODE_FILE}.bak" "$CODE_FILE"
echo "=== Pushed $CONFIG_NAME as $KERNEL_ID ==="
