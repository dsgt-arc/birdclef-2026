#!/usr/bin/env bash
# Push Kaggle kernel with config override for experiment 023.
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
META="kernel-metadata-ensemble.json"
CODE_FILE="submission_ensemble.py"

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

# Convert YAML config to Python dict literal (JSON subset)
OVERRIDE=$(python3 -c "
import yaml, json
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
# Remove metadata keys, keep only config values
cfg.pop('name', None)
cfg.pop('description', None)
print(json.dumps(cfg))
")

echo "=== Pushing kernel: $CONFIG_NAME ==="
echo "Config: $CONFIG_FILE"
echo "Override: $OVERRIDE"

# Inject override into kernel
cp "$CODE_FILE" "${CODE_FILE}.bak"
sed -i "s|^CFG_OVERRIDE = {}.*|CFG_OVERRIDE = $OVERRIDE  # from configs/${CONFIG_NAME}.yaml|" "$CODE_FILE"

# Push
cp "$META" kernel-metadata.json
kaggle kernels push -p .
rm kernel-metadata.json

# Restore original script
mv "${CODE_FILE}.bak" "$CODE_FILE"
echo "=== Pushed $CONFIG_NAME, $CODE_FILE restored ==="
