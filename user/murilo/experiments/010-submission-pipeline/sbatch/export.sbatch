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
# Model naming: {encoder}-{year}[-{variant}]
# Prefer using run.sh instead of calling this directly.
#
# Usage (via run.sh):
#   bash sbatch/run.sh export --model protoclr-2026          # 2026 80/10/10
#   bash sbatch/run.sh export --model protoclr-2026-fulldata # 2026 fulldata
#   bash sbatch/run.sh export --model all-2026-fulldata      # all 3, fulldata
#
# Direct usage:
#   MODEL=protoclr-2026-fulldata sbatch sbatch/export.sbatch
#   MODEL=all-2026 sbatch sbatch/export.sbatch

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

export_eat_2025() {
    echo "=== Exporting EAT (2025-trained, exp 006) ==="
    local CKPT="${CHECKPOINTS_BASE}/006-lora-finetune/checkpoints/eat-lora-attn-mlphead/best-epochepoch=16.ckpt"
    python scripts/export_eat.py \
        --checkpoint "$CKPT" \
        --output-dir artifacts/weights/eat-2025/
}

export_convnext_2025() {
    echo "=== Exporting ConvNeXt (2025-trained, exp 007) ==="
    local CKPT="${CHECKPOINTS_BASE}/007-convnext-finetune/checkpoints/convnext-lora-mlp-mlphead/best-epochepoch=12.ckpt"
    python scripts/export_convnext.py \
        --checkpoint "$CKPT" \
        --output-dir artifacts/weights/convnext-2025/
}

export_protoclr_2025() {
    echo "=== Exporting ProtoCLR (2025-trained, exp 009) ==="
    local CKPT="${CHECKPOINTS_BASE}/009-protoclr-finetune/checkpoints/protoclr-lora-attn/best-epochepoch=20.ckpt"
    python scripts/export_protoclr.py \
        --checkpoint "$CKPT" \
        --output-dir artifacts/weights/protoclr-2025/
}

# ============================================================================
# 2026-trained model exports (exp 011 checkpoints)
# ============================================================================

TRAIN_CSV_2026="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/train.csv"
EXP011_DIR="$(dirname "$EXPERIMENT_DIR")/011-lora-baseline-2026"

export_protoclr_2026() {
    echo "=== Exporting ProtoCLR (2026-trained, 80/10/10) ==="
    local CKPT="${CHECKPOINTS_BASE}/011-lora-baseline-2026/checkpoints/2026-protoclr-lora-attn/best-epochepoch=21.ckpt"
    python scripts/export_protoclr.py \
        --checkpoint "$CKPT" \
        --config "${EXP011_DIR}/config/protoclr_lora_attn.yaml" \
        --train-csv "$TRAIN_CSV_2026" \
        --output-dir artifacts/weights/protoclr-2026/
}

export_eat_2026() {
    echo "=== Exporting EAT (2026-trained, 80/10/10) ==="
    local CKPT="${CHECKPOINTS_BASE}/011-lora-baseline-2026/checkpoints/2026-eat-lora-attn-mlphead/best-epochepoch=17.ckpt"
    python scripts/export_eat.py \
        --checkpoint "$CKPT" \
        --config "${EXP011_DIR}/config/eat_lora_attn_mlphead.yaml" \
        --train-csv "$TRAIN_CSV_2026" \
        --output-dir artifacts/weights/eat-2026/
}

export_convnext_2026() {
    echo "=== Exporting ConvNeXt (2026-trained, 80/10/10) ==="
    local CKPT="${CHECKPOINTS_BASE}/011-lora-baseline-2026/checkpoints/2026-convnext-lora-mlp-mlphead/best-epochepoch=09.ckpt"
    python scripts/export_convnext.py \
        --checkpoint "$CKPT" \
        --config "${EXP011_DIR}/config/convnext_lora_mlp_mlphead.yaml" \
        --train-csv "$TRAIN_CSV_2026" \
        --output-dir artifacts/weights/convnext-2026/
}

# ============================================================================
# 2026 fulldata model exports (exp 011 --train-all checkpoints)
# ============================================================================

# Helper: find the most recent last*.ckpt in a checkpoint directory.
# PL may create last.ckpt or last-v1.ckpt depending on save timing.
find_last_ckpt() {
    local ckpt_dir="$1"
    local ckpt
    ckpt="$(ls -t "$ckpt_dir"/last*.ckpt 2>/dev/null | head -1)"
    if [ -z "$ckpt" ]; then
        echo "ERROR: No last*.ckpt found in $ckpt_dir" >&2
        exit 1
    fi
    echo "$ckpt"
}

export_protoclr_2026_fulldata() {
    echo "=== Exporting ProtoCLR (2026-trained, fulldata) ==="
    local CKPT_DIR="${CHECKPOINTS_BASE}/011-lora-baseline-2026/checkpoints/2026-protoclr-lora-attn-fulldata"
    local CKPT
    CKPT="$(find_last_ckpt "$CKPT_DIR")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_protoclr.py \
        --checkpoint "$CKPT" \
        --config "${EXP011_DIR}/config/protoclr_lora_attn.yaml" \
        --train-csv "$TRAIN_CSV_2026" \
        --output-dir artifacts/weights/protoclr-2026-fulldata/
}

export_eat_2026_fulldata() {
    echo "=== Exporting EAT (2026-trained, fulldata) ==="
    local CKPT_DIR="${CHECKPOINTS_BASE}/011-lora-baseline-2026/checkpoints/2026-eat-lora-attn-mlphead-fulldata"
    local CKPT
    CKPT="$(find_last_ckpt "$CKPT_DIR")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_eat.py \
        --checkpoint "$CKPT" \
        --config "${EXP011_DIR}/config/eat_lora_attn_mlphead.yaml" \
        --train-csv "$TRAIN_CSV_2026" \
        --output-dir artifacts/weights/eat-2026-fulldata/
}

export_convnext_2026_fulldata() {
    echo "=== Exporting ConvNeXt (2026-trained, fulldata) ==="
    local CKPT_DIR="${CHECKPOINTS_BASE}/011-lora-baseline-2026/checkpoints/2026-convnext-lora-mlp-mlphead-fulldata"
    local CKPT
    CKPT="$(find_last_ckpt "$CKPT_DIR")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_convnext.py \
        --checkpoint "$CKPT" \
        --config "${EXP011_DIR}/config/convnext_lora_mlp_mlphead.yaml" \
        --train-csv "$TRAIN_CSV_2026" \
        --output-dir artifacts/weights/convnext-2026-fulldata/
}

# ============================================================================
# 2026-trained soundscape model exports (exp 012 checkpoints, 234 classes)
# ============================================================================

TAXONOMY_CSV_2026="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/taxonomy.csv"
EXP012_DIR="$(dirname "$EXPERIMENT_DIR")/012-soundscape-finetune"

export_protoclr_2026_soundscape() {
    echo "=== Exporting ProtoCLR (2026-soundscape, 234 classes) ==="
    local CKPT="${CHECKPOINTS_BASE}/012-soundscape-finetune/checkpoints/2026-protoclr-soundscape/best-epochepoch=11.ckpt"
    python scripts/export_protoclr.py \
        --checkpoint "$CKPT" \
        --config "${EXP012_DIR}/config/protoclr_soundscape.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/protoclr-2026-soundscape/
}

export_eat_2026_soundscape() {
    echo "=== Exporting EAT (2026-soundscape, 234 classes) ==="
    local CKPT_DIR="${CHECKPOINTS_BASE}/012-soundscape-finetune/checkpoints/2026-eat-soundscape"
    local CKPT
    CKPT="$(ls -t "$CKPT_DIR"/best-epoch*.ckpt 2>/dev/null | head -1)"
    if [ -z "$CKPT" ]; then
        echo "ERROR: No best-epoch*.ckpt found in $CKPT_DIR" >&2
        exit 1
    fi
    echo "Using checkpoint: $CKPT"
    python scripts/export_eat.py \
        --checkpoint "$CKPT" \
        --config "${EXP012_DIR}/config/eat_soundscape.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/eat-2026-soundscape/
}

# ============================================================================
# 2026-trained student model exports (exp 015 checkpoints, 234 classes)
# ============================================================================

EXP015_DIR="$(dirname "$EXPERIMENT_DIR")/015-student-training"

export_protoclr_2026_student_v1() {
    echo "=== Exporting ProtoCLR (2026-student-v1, 234 classes) ==="
    local CKPT="${CHECKPOINTS_BASE}/015-student-training/checkpoints/2026-protoclr-student-v1/best-epochepoch=05.ckpt"
    python scripts/export_protoclr.py \
        --checkpoint "$CKPT" \
        --config "${EXP015_DIR}/config/protoclr_student.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/protoclr-2026-student-v1/
}

export_eat_2026_student_v1() {
    echo "=== Exporting EAT (2026-student-v1, 234 classes) ==="
    local CKPT="${CHECKPOINTS_BASE}/015-student-training/checkpoints/2026-eat-student-v1/best-epochepoch=02.ckpt"
    python scripts/export_eat.py \
        --checkpoint "$CKPT" \
        --config "${EXP015_DIR}/config/eat_student.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/eat-2026-student-v1/
}

# ============================================================================
# 2026 SED model exports (exp 016 checkpoints, 234 classes)
# ============================================================================

EXP016_DIR="$(dirname "$EXPERIMENT_DIR")/016-sed-architecture"

export_effnet_b0_sed_v1() {
    echo "=== Exporting EfficientNet-B0 SED (2026, 234 classes) ==="
    local CKPT="${CHECKPOINTS_BASE}/016-sed-architecture/checkpoints/2026-effnet-b0-sed-v1/best-epochepoch=24.ckpt"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP016_DIR}/config/efficientnet_b0_sed.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v1/
}

export_convnext_2026_soundscape() {
    echo "=== Exporting ConvNeXt (2026-soundscape, 234 classes) ==="
    local CKPT_DIR="${CHECKPOINTS_BASE}/012-soundscape-finetune/checkpoints/2026-convnext-soundscape"
    local CKPT
    CKPT="$(ls -t "$CKPT_DIR"/best-epoch*.ckpt 2>/dev/null | head -1)"
    if [ -z "$CKPT" ]; then
        echo "ERROR: No best-epoch*.ckpt found in $CKPT_DIR" >&2
        exit 1
    fi
    echo "Using checkpoint: $CKPT"
    python scripts/export_convnext.py \
        --checkpoint "$CKPT" \
        --config "${EXP012_DIR}/config/convnext_soundscape.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/convnext-2026-soundscape/
}

# ============================================================================
# Build species mappings and run exports
# ============================================================================

case "$MODEL" in
    eat-2025|convnext-2025|protoclr-2025|all-2025)
        echo "=== Building species mapping (2025→2026, 41 overlap) ==="
        python scripts/build_species_mapping.py \
            --output artifacts/weights/species_mapping.json
        echo ""
        ;;
esac

case "$MODEL" in
    eat-2026|convnext-2026|protoclr-2026|all-2026|\
    eat-2026-fulldata|convnext-2026-fulldata|protoclr-2026-fulldata|all-2026-fulldata)
        echo "=== Building species mapping (2026→2026, 206 overlap) ==="
        python scripts/build_species_mapping.py \
            --train-csv "$TRAIN_CSV_2026" \
            --output artifacts/weights/species_mapping_2026.json
        echo ""
        ;;
esac

case "$MODEL" in
    eat-2026-soundscape|convnext-2026-soundscape|protoclr-2026-soundscape|all-2026-soundscape|\
    protoclr-2026-student-v1|eat-2026-student-v1|all-2026-student|\
    effnet-b0-sed-v1|all-2026-sed)
        echo "=== Building species mapping (2026→2026, 234/234 identity) ==="
        python scripts/build_species_mapping.py \
            --train-csv "$TAXONOMY_CSV_2026" \
            --output artifacts/weights/species_mapping_2026_234.json
        echo ""
        ;;
esac

case "$MODEL" in
    eat-2025)                export_eat_2025 ;;
    convnext-2025)           export_convnext_2025 ;;
    protoclr-2025)           export_protoclr_2025 ;;
    eat-2026)                export_eat_2026 ;;
    convnext-2026)           export_convnext_2026 ;;
    protoclr-2026)           export_protoclr_2026 ;;
    eat-2026-fulldata)       export_eat_2026_fulldata ;;
    convnext-2026-fulldata)  export_convnext_2026_fulldata ;;
    protoclr-2026-fulldata)  export_protoclr_2026_fulldata ;;
    all-2025)
        export_eat_2025
        echo ""
        export_convnext_2025
        echo ""
        export_protoclr_2025
        ;;
    all-2026)
        export_eat_2026
        echo ""
        export_convnext_2026
        echo ""
        export_protoclr_2026
        ;;
    all-2026-fulldata)
        export_eat_2026_fulldata
        echo ""
        export_convnext_2026_fulldata
        echo ""
        export_protoclr_2026_fulldata
        ;;
    eat-2026-soundscape)       export_eat_2026_soundscape ;;
    convnext-2026-soundscape)  export_convnext_2026_soundscape ;;
    protoclr-2026-soundscape)  export_protoclr_2026_soundscape ;;
    all-2026-soundscape)
        export_eat_2026_soundscape
        echo ""
        export_convnext_2026_soundscape
        echo ""
        export_protoclr_2026_soundscape
        ;;
    protoclr-2026-student-v1)  export_protoclr_2026_student_v1 ;;
    eat-2026-student-v1)       export_eat_2026_student_v1 ;;
    all-2026-student)
        export_protoclr_2026_student_v1
        echo ""
        export_eat_2026_student_v1
        ;;
    effnet-b0-sed-v1)          export_effnet_b0_sed_v1 ;;
    all-2026-sed)
        export_effnet_b0_sed_v1
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
