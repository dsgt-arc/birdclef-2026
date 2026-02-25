#!/bin/bash
#SBATCH --job-name=012-export-onnx
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -t30 -qinferno
#SBATCH --output=logs/export-onnx-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv

cd $repo_dir/user/anthony/experiments/012-efficientnet-linear-probe

# Find best checkpoint (highest val/cmap) or accept as $1
# Note: finetune-{epoch}-{val/cmap} pattern creates subdirs due to / in metric name
CKPT_PATH=${1:-$(find logs/finetune/checkpoints -name 'finetune-*.ckpt' -not -name 'last.ckpt' 2>/dev/null | sort -t'-' -k3 -rn | head -1)}
if [ -z "$CKPT_PATH" ]; then
    # Fall back to last.ckpt
    CKPT_PATH="logs/finetune/checkpoints/last.ckpt"
fi
if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: No checkpoint found at $CKPT_PATH. Provide path as argument or run finetuning first."
    exit 1
fi

METADATA_CSV=$scratch_dir/data/birdclef-2025/train.csv

echo "=== ONNX Export (EfficientNet-B1) ==="
echo "Checkpoint: $CKPT_PATH"
echo "Config: configs/finetune.yaml"
echo "Metadata: $METADATA_CSV"
echo "Output: artifacts/weights/"

uv run --project $repo_dir/user/anthony python scripts/export_onnx.py \
    --checkpoint "$CKPT_PATH" \
    --config configs/finetune.yaml \
    --metadata-csv "$METADATA_CSV" \
    --output-dir artifacts/weights/

echo ""
echo "=== Export complete ==="
ls -lh artifacts/weights/
