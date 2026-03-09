#!/bin/bash
#SBATCH --job-name=014-export-onnx
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -t30 -qinferno
#SBATCH --output=logs/export-onnx-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv

cd $repo_dir/user/anthony/experiments/014-efficientnet-soft-labels

# Usage: sbatch export-onnx.sbatch <checkpoint_path> [run_name]
CKPT_PATH=${1:?Usage: sbatch export-onnx.sbatch <checkpoint_path> [run_name]}
RUN_NAME=${2:-default}
OUTPUT_DIR="artifacts/weights-${RUN_NAME}"

if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: No checkpoint found at $CKPT_PATH"
    exit 1
fi

METADATA_CSV=$scratch_dir/data/birdclef-2025/train.csv

echo "=== ONNX Export (EfficientNet-B1 Soft Labels — $RUN_NAME) ==="
echo "Checkpoint: $CKPT_PATH"
echo "Config: configs/finetune-soft.yaml"
echo "Metadata: $METADATA_CSV"
echo "Output: $OUTPUT_DIR"

uv run --project $repo_dir/user/anthony python scripts/export_onnx.py \
    --checkpoint "$CKPT_PATH" \
    --config configs/finetune-soft.yaml \
    --metadata-csv "$METADATA_CSV" \
    --output-dir "$OUTPUT_DIR"

# Also copy inference script into weights dir
cp scripts/kaggle_inference.py "$OUTPUT_DIR/"

echo ""
echo "=== Export complete ==="
ls -lh "$OUTPUT_DIR"
