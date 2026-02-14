#!/usr/bin/env bash
# Export model weights from Lightning checkpoint to standalone model.pt for Kaggle
#
# Usage: bash scripts/export_weights.sh <lightning_checkpoint.ckpt> <output_model.pt>

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <lightning_checkpoint.ckpt> <output_model.pt>"
    exit 1
fi

CKPT_PATH="$1"
OUTPUT_PATH="$2"

python -c "
import torch
import sys

ckpt = torch.load('${CKPT_PATH}', map_location='cpu', weights_only=False)
state_dict = ckpt.get('state_dict', ckpt)

# Strip 'model.' prefix from Lightning state dict
model_state = {}
for key, value in state_dict.items():
    if key.startswith('model.'):
        model_state[key[len('model.'):]] = value
    else:
        model_state[key] = value

torch.save(model_state, '${OUTPUT_PATH}')
print(f'Exported {len(model_state)} keys to ${OUTPUT_PATH}')
"

echo "Done: $OUTPUT_PATH"
