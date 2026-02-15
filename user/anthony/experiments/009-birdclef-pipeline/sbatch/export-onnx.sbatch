#!/bin/bash
#SBATCH --job-name=009-export-onnx
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-rtx6000
#SBATCH -t60 -qinferno
#SBATCH --output=logs/export-onnx-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv

cd $repo_dir/user/anthony/experiments/009-birdclef-pipeline

module load cuda/12.1.1

# Find best checkpoint (highest val/cmap)
CKPT_PATH=${1:-$(ls -t logs/finetune/checkpoints/finetune-*.ckpt 2>/dev/null | head -1)}
if [ -z "$CKPT_PATH" ]; then
    echo "ERROR: No checkpoint found. Provide path as argument or run finetuning first."
    exit 1
fi

METADATA_CSV=$scratch_dir/data/birdclef-2025/train.csv

echo "=== ONNX Export ==="
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
