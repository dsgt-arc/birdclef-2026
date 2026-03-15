#!/bin/bash
#SBATCH --job-name=exp010-export
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Experiment 010: ONNX Export for BirdCLEF 2026 Submission
# GPU needed for model construction (HuggingFace) and ONNX tracing.
#
# Usage:
#   cd user/murilo/experiments/010-submission-pipeline
#   sbatch sbatch/export.sbatch              # export all 3 models
#   MODEL=eat sbatch sbatch/export.sbatch    # export specific model

set -eu

# Which model(s) to export
MODEL="${MODEL:-all}"

# ============================================================================
# Paths
# ============================================================================

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    EXPERIMENT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
fi

USER_DIR="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
REPO_DIR="$(dirname "$(dirname "$USER_DIR")")"
PROJECT_NAME="${PROJECT_NAME:-ps-clef2026_birdclef-0}"
CHECKPOINTS_BASE="/storage/project/${PROJECT_NAME}/mgustineli3/experiments"

echo "=============================================="
echo "Job: Experiment 010 - ONNX Export"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Model: ${MODEL}"
echo "Experiment: ${EXPERIMENT_DIR}"
echo ""
nvidia-smi
echo ""

# ============================================================================
# Setup venv in TMPDIR
# ============================================================================

export UV_CACHE_DIR="$TMPDIR/uv-cache"
JOB_VENV_DIR="$TMPDIR/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"

echo "Setting up venv at $JOB_VENV..."
mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

uv pip install -e "$EXPERIMENT_DIR"
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo "Venv ready!"
echo ""

python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
echo ""

# HuggingFace cache
export HF_HOME="/storage/project/${PROJECT_NAME}/mgustineli3/cache/huggingface"
mkdir -p "$HF_HOME"

cd "$EXPERIMENT_DIR"

# ============================================================================
# Run exports
# ============================================================================

export_eat() {
    echo "=== Exporting EAT ==="
    local CKPT="${CHECKPOINTS_BASE}/006-lora-finetune/checkpoints/eat-lora-attn-mlphead/best-epochepoch=16.ckpt"
    python scripts/export_eat.py \
        --checkpoint "$CKPT" \
        --output-dir artifacts/weights/eat/
}

export_convnext() {
    echo "=== Exporting ConvNeXt ==="
    local CKPT="${CHECKPOINTS_BASE}/007-convnext-finetune/checkpoints/convnext-lora-mlp-mlphead/best-epochepoch=12.ckpt"
    python scripts/export_convnext.py \
        --checkpoint "$CKPT" \
        --output-dir artifacts/weights/convnext/
}

export_protoclr() {
    echo "=== Exporting ProtoCLR ==="
    local CKPT="${CHECKPOINTS_BASE}/009-protoclr-finetune/checkpoints/protoclr-lora-attn/best-epochepoch=20.ckpt"
    python scripts/export_protoclr.py \
        --checkpoint "$CKPT" \
        --output-dir artifacts/weights/protoclr/
}

# ============================================================================
# 2026-trained model exports (exp 011 checkpoints)
# ============================================================================

TRAIN_CSV_2026="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/train.csv"
EXP011_DIR="$(dirname "$EXPERIMENT_DIR")/011-lora-baseline-2026"

export_protoclr_2026() {
    echo "=== Exporting ProtoCLR (2026-trained) ==="
    local CKPT="${CHECKPOINTS_BASE}/011-lora-baseline-2026/checkpoints/2026-protoclr-lora-attn/best-epochepoch=21.ckpt"
    python scripts/export_protoclr.py \
        --checkpoint "$CKPT" \
        --config "${EXP011_DIR}/config/protoclr_lora_attn.yaml" \
        --train-csv "$TRAIN_CSV_2026" \
        --output-dir artifacts/weights/protoclr-2026/
}

export_eat_2026() {
    echo "=== Exporting EAT (2026-trained) ==="
    local CKPT="${CHECKPOINTS_BASE}/011-lora-baseline-2026/checkpoints/2026-eat-lora-attn-mlphead/best-epochepoch=17.ckpt"
    python scripts/export_eat.py \
        --checkpoint "$CKPT" \
        --config "${EXP011_DIR}/config/eat_lora_attn_mlphead.yaml" \
        --train-csv "$TRAIN_CSV_2026" \
        --output-dir artifacts/weights/eat-2026/
}

export_convnext_2026() {
    echo "=== Exporting ConvNeXt (2026-trained) ==="
    local CKPT="${CHECKPOINTS_BASE}/011-lora-baseline-2026/checkpoints/2026-convnext-lora-mlp-mlphead/best-epochepoch=09.ckpt"
    python scripts/export_convnext.py \
        --checkpoint "$CKPT" \
        --config "${EXP011_DIR}/config/convnext_lora_mlp_mlphead.yaml" \
        --train-csv "$TRAIN_CSV_2026" \
        --output-dir artifacts/weights/convnext-2026/
}

# ============================================================================
# Build species mappings and run exports
# ============================================================================

case "$MODEL" in
    # 2025-trained models
    eat|convnext|protoclr|all)
        echo "=== Building species mapping (2025→2026) ==="
        python scripts/build_species_mapping.py \
            --output artifacts/weights/species_mapping.json
        echo ""
        ;;
esac

case "$MODEL" in
    # 2026-trained models
    protoclr-2026|eat-2026|convnext-2026|all-2026)
        echo "=== Building species mapping (2026→2026) ==="
        python scripts/build_species_mapping.py \
            --train-csv "$TRAIN_CSV_2026" \
            --output artifacts/weights/species_mapping_2026.json
        echo ""
        ;;
esac

case "$MODEL" in
    eat)            export_eat ;;
    convnext)       export_convnext ;;
    protoclr)       export_protoclr ;;
    protoclr-2026)  export_protoclr_2026 ;;
    eat-2026)       export_eat_2026 ;;
    convnext-2026)  export_convnext_2026 ;;
    all)
        export_eat
        echo ""
        export_convnext
        echo ""
        export_protoclr
        ;;
    all-2026)
        export_protoclr_2026
        echo ""
        export_eat_2026
        echo ""
        export_convnext_2026
        ;;
    *)
        echo "Unknown model: $MODEL"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Export complete! $(date)"
echo "=============================================="
echo ""
echo "Artifacts:"
find artifacts/weights -type f -exec ls -lh {} \;
