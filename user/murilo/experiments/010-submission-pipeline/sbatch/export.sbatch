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
    echo "=== Exporting EfficientNet-B0 SED v1 (2026, 234 classes) ==="
    local CKPT="${CHECKPOINTS_BASE}/016-sed-architecture/checkpoints/2026-effnet-b0-sed-v1/best-epochepoch=24.ckpt"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP016_DIR}/config/efficientnet_b0_sed.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v1/
}

export_effnet_b0_sed_v2() {
    echo "=== Exporting EfficientNet-B0 SED v2 (NS-JFT + CE, 234 classes) ==="
    local CKPT="${CHECKPOINTS_BASE}/016-sed-architecture/checkpoints/2026-effnet-b0-sed-v2/best-epochepoch=24.ckpt"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP016_DIR}/config/efficientnet_b0_sed_v2.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v2/
}

export_effnet_b0_sed_v3a() {
    echo "=== Exporting EfficientNet-B0 SED v3a (TF weights, 234 classes) ==="
    local CKPT="${CHECKPOINTS_BASE}/016-sed-architecture/checkpoints/2026-effnet-b0-sed-v3a/best-epochepoch=25.ckpt"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP016_DIR}/config/efficientnet_b0_sed_v3a.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v3a/
}

export_effnet_b0_sed_v3b() {
    echo "=== Exporting EfficientNet-B0 SED v3b (raw waveform mixup, 234 classes) ==="
    local CKPT="${CHECKPOINTS_BASE}/016-sed-architecture/checkpoints/2026-effnet-b0-sed-v3b/best-epochepoch=25.ckpt"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP016_DIR}/config/efficientnet_b0_sed_v3b.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v3b/
}

export_effnet_b0_sed_v3c() {
    echo "=== Exporting EfficientNet-B0 SED v3c (CE loss, 234 classes) ==="
    local CKPT="${CHECKPOINTS_BASE}/016-sed-architecture/checkpoints/2026-effnet-b0-sed-v3c/best-epochepoch=25.ckpt"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP016_DIR}/config/efficientnet_b0_sed_v3c.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v3c/
}

export_effnet_b0_sed_v3d() {
    echo "=== Exporting EfficientNetV2-B0 SED v3d (V2 architecture, 234 classes) ==="
    local CKPT_DIR="${CHECKPOINTS_BASE}/016-sed-architecture/checkpoints/2026-effnet-b0-sed-v3d"
    local CKPT
    CKPT="$(ls -t "$CKPT_DIR"/best-epoch*.ckpt 2>/dev/null | head -1)"
    if [ -z "$CKPT" ]; then
        echo "ERROR: No best-epoch*.ckpt found in $CKPT_DIR" >&2
        exit 1
    fi
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP016_DIR}/config/efficientnet_b0_sed_v3d.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v3d/
}

# ============================================================================
# Exp 017: Mel overhaul SED model exports (n_fft=4096, n_mels=224)
# ============================================================================

EXP017_DIR="$(dirname "$EXPERIMENT_DIR")/017-mel-overhaul"
EXP019_DIR="$(dirname "$EXPERIMENT_DIR")/019-sed-ablations"

export_effnet_b0_sed_v4a() {
    echo "=== Exporting EfficientNet-B0 SED v4a (mel overhaul, 234 classes) ==="
    local CKPT="${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-effnet-b0-sed-v4a/best-epochepoch=24.ckpt"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP017_DIR}/config/efficientnet_b0_sed_v4a.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v4a/
}

export_effnet_b0_sed_v4b() {
    echo "=== Exporting EfficientNet-B0 SED v4b (AmplitudeToDB norm, 234 classes) ==="
    local CKPT="${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-effnet-b0-sed-v4b/best-epochepoch=30.ckpt"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP017_DIR}/config/efficientnet_b0_sed_v4b.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v4b/
}

export_effnet_b0_sed_v4c() {
    echo "=== Exporting EfficientNet-B0 SED v4c (AttBlockV2, 234 classes) ==="
    local CKPT="${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-effnet-b0-sed-v4c/best-epochepoch=15.ckpt"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP017_DIR}/config/efficientnet_b0_sed_v4c.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v4c/
}

export_effnet_b0_sed_v4d() {
    echo "=== Exporting EfficientNet-B0 SED v4d (ASL loss, 234 classes) ==="
    local CKPT="${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-effnet-b0-sed-v4d/best-epochepoch=16.ckpt"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP017_DIR}/config/efficientnet_b0_sed_v4d.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v4d/
}

export_effnet_b0_sed_v4e() {
    echo "=== Exporting EfficientNet-B0 SED v4e (StochasticDepth, 234 classes) ==="
    local CKPT="${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-effnet-b0-sed-v4e/best-epochepoch=29.ckpt"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP017_DIR}/config/efficientnet_b0_sed_v4e.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v4e/
}

export_effnet_b0_sed_v4f() {
    echo "=== Exporting EfficientNet-B0 SED v4f (AmplitudeToDB + StochasticDepth, 234 classes) ==="
    local CKPT_DIR="${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-effnet-b0-sed-v4f"
    local CKPT
    CKPT="$(ls -t "$CKPT_DIR"/best-epoch*.ckpt 2>/dev/null | head -1)"
    if [ -z "$CKPT" ]; then
        echo "ERROR: No best-epoch*.ckpt found in $CKPT_DIR" >&2
        exit 1
    fi
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP017_DIR}/config/efficientnet_b0_sed_v4f.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v4f/
}

export_effnet_b0_sed_v4b_fulldata() {
    echo "=== Exporting EfficientNet-B0 SED v4b-fulldata (AmplitudeToDB, fulldata, 234 classes) ==="
    local CKPT_DIR="${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-effnet-b0-sed-v4b-fulldata"
    local CKPT
    # Use last.ckpt for fulldata (no "best" since no val monitoring)
    CKPT="$(ls -t "$CKPT_DIR"/last*.ckpt 2>/dev/null | head -1)"
    if [ -z "$CKPT" ]; then
        echo "ERROR: No last*.ckpt found in $CKPT_DIR" >&2
        exit 1
    fi
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP017_DIR}/config/efficientnet_b0_sed_v4b_fulldata.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v4b-fulldata/
}

export_effnet_b0_sed_v4g() {
    echo "=== Exporting EfficientNet-B0 SED v4g (AttBlockV2 + BCE, 234 classes) ==="
    local CKPT_DIR="${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-effnet-b0-sed-v4g"
    local CKPT
    CKPT="$(ls -t "$CKPT_DIR"/best-epoch*.ckpt 2>/dev/null | head -1)"
    if [ -z "$CKPT" ]; then
        echo "ERROR: No best-epoch*.ckpt found in $CKPT_DIR" >&2
        exit 1
    fi
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP017_DIR}/config/efficientnet_b0_sed_v4g.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v4g/
}

export_effnet_b0_sed_v4h() {
    echo "=== Exporting EfficientNet-B0 SED v4h (AttBlockV2 + ASL, 234 classes) ==="
    local CKPT_DIR="${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-effnet-b0-sed-v4h"
    local CKPT
    CKPT="$(ls -t "$CKPT_DIR"/best-epoch*.ckpt 2>/dev/null | head -1)"
    if [ -z "$CKPT" ]; then
        echo "ERROR: No best-epoch*.ckpt found in $CKPT_DIR" >&2
        exit 1
    fi
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP017_DIR}/config/efficientnet_b0_sed_v4h.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v4h/
}

export_effnet_b0_sed_v4j() {
    echo "=== Exporting EfficientNet-B0 SED v4j (waveform MixUp, 234 classes) ==="
    local CKPT="${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-effnet-b0-sed-v4j/best-epochepoch=34.ckpt"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP017_DIR}/config/efficientnet_b0_sed_v4j.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v4j/
}

_find_best_ckpt() {
    local DIR="$1"
    local CKPT
    CKPT="$(ls -t "$DIR"/best-epoch*.ckpt 2>/dev/null | head -1)"
    if [ -z "$CKPT" ]; then
        echo "ERROR: No best-epoch*.ckpt found in $DIR" >&2
        exit 1
    fi
    echo "$CKPT"
}

export_effnet_b0_sed_v5a() {
    echo "=== Exporting EfficientNet-B0 SED v5a (BCE loss, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-effnet-b0-sed-v5a")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP019_DIR}/config/effnet_b0_sed_v5a.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v5a/
}

export_effnet_b0_sed_v5b() {
    echo "=== Exporting EfficientNet-B0 SED v5b (stronger MixUp, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-effnet-b0-sed-v5b")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP019_DIR}/config/effnet_b0_sed_v5b.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v5b/
}

export_effnet_b0_sed_v5c() {
    echo "=== Exporting EfficientNet-B0 SED v5c (low weight_decay, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-effnet-b0-sed-v5c")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP019_DIR}/config/effnet_b0_sed_v5c.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v5c/
}

export_effnet_b0_sed_v5d() {
    echo "=== Exporting EfficientNet-B0 SED v5d (LSEHead, 234 classes) ==="
    local CKPT="${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-effnet-b0-sed-v5d/best-epochepoch=21.ckpt"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP019_DIR}/config/effnet_b0_sed_v5d.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v5d/
}

export_hgnet_b0_sed_v5e() {
    echo "=== Exporting HGNetV2-B0 SED v5e (HGNet backbone, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5e")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP019_DIR}/config/hgnet_b0_sed_v5e.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v5e/
}

export_effnet_b0_sed_v5f() {
    echo "=== Exporting EfficientNet-B0 SED v5f (ttahara recipe, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-effnet-b0-sed-v5f")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP019_DIR}/config/effnet_b0_sed_v5f.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v5f/
}

export_effnet_b0_sed_v5n() {
    echo "=== Exporting EfficientNet-B0 SED v5n (no projection, v5j recipe, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-effnet-b0-sed-v5n")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP019_DIR}/config/effnet_b0_sed_v5n.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v5n/
}

export_effnet_b0_sed_v5m() {
    echo "=== Exporting EfficientNet-B0 SED v5m (projection 320→1024, v5j recipe, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-effnet-b0-sed-v5m")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP019_DIR}/config/effnet_b0_sed_v5m.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v5m/
}

export_hgnet_b0_sed_v5h() {
    echo "=== Exporting HGNetV2-B0 SED v5h (v5g + n_fft=4096, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5h")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP019_DIR}/config/hgnet_b0_sed_v5h.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v5h/
}

export_hgnet_b0_sed_v5i() {
    echo "=== Exporting HGNetV2-B0 SED v5i (v5g + 10s windows, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5i")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP019_DIR}/config/hgnet_b0_sed_v5i.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v5i/
}

export_hgnet_b0_sed_v5l() {
    echo "=== Exporting HGNetV2-B0 SED v5l (v5j + n_fft=4096, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5l")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP019_DIR}/config/hgnet_b0_sed_v5l.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v5l/
}

export_hgnet_b0_sed_v5k() {
    echo "=== Exporting HGNetV2-B0 SED v5k (v5g + 3-channel input, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5k")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP019_DIR}/config/hgnet_b0_sed_v5k.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v5k/
}

export_hgnet_b0_sed_v5j() {
    echo "=== Exporting HGNetV2-B0 SED v5j (v5g + AmplitudeToDB, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5j")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP019_DIR}/config/hgnet_b0_sed_v5j.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v5j/
}

export_hgnet_b0_sed_v5j_resubmit() {
    # Re-export of the ORIGINAL v5j checkpoint (epoch 19, LB 0.906 from exp 019)
    # using TODAY's submission.py kernel. Diagnostic for separating training-pipeline
    # drift from inference-pipeline drift after v5j_control_v2 regressed to 0.896 LB.
    # Same checkpoint and config as `hgnet-b0-sed-v5j`, different output dir to keep
    # the original kernel intact.
    echo "=== Exporting HGNetV2-B0 SED v5j resubmit (original ckpt, fresh kernel) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5j")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP019_DIR}/config/hgnet_b0_sed_v5j.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v5j-resubmit/
}

export_hgnet_b0_sed_v5j_finetune_2026() {
    echo "=== Exporting HGNetV2-B0 SED v5j fine-tune 2026 (2025-pretrained, exp 029) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/029-2025-pretrain/checkpoints/2026-exp029-v5j-finetune-2026")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP029_DIR}/configs/v5j_finetune_2026.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v5j-finetune-2026/
}

export_hgnet_b0_sed_v5j_ft_splitlr() {
    echo "=== Exporting HGNetV2-B0 SED v5j fine-tune 2026 split-LR retry (exp 029 path A) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/029-2025-pretrain/checkpoints/2026-exp029-v5j-finetune-2026-split-lr")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP029_DIR}/configs/v5j_finetune_2026_split_lr.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v5j-ft-splitlr/
}

# ============================================================================
# Exp 030: Sydorskyi-faithful MixUp (HGNet 2x2 + EffNetV2-B0 2x2)
# ============================================================================

export_hgnet_b0_sed_v5j_control() {
    # v2: dual_validation removed (v1 hit -0.009 LB from focal-only val
    # checkpoint selection). Run name has -v2 suffix; checkpoint dir is
    # 2026-exp030-v5j-control-v2/.
    echo "=== Exporting Exp 030 v5j_control v2 (HGNetV2-B0, no dual_val) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/030-sydorskyi-mixup/checkpoints/2026-exp030-v5j-control-v2")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP030_DIR}/configs/v5j_control.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v5j-control/
}

export_hgnet_b0_sed_v5j_sydorskyi() {
    # v2: dual_validation removed. Run name has -v2 suffix.
    echo "=== Exporting Exp 030 v5j_sydorskyi v2 (HGNetV2-B0 + Sydorskyi wave-mixup, no dual_val) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/030-sydorskyi-mixup/checkpoints/2026-exp030-v5j-sydorskyi-v2")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP030_DIR}/configs/v5j_sydorskyi.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v5j-sydorskyi/
}

export_effnetv2_b0_sed_control() {
    # v2: dual_validation removed. Run name has -v2 suffix.
    echo "=== Exporting Exp 030 effnetv2b0_control v2 (EfficientNetV2-B0, no dual_val) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/030-sydorskyi-mixup/checkpoints/2026-exp030-effnetv2b0-control-v2")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP030_DIR}/configs/effnetv2b0_control.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnetv2-b0-sed-control/
}

export_effnetv2_b0_sed_sydorskyi() {
    # v2: dual_validation removed. Run name has -v2 suffix.
    echo "=== Exporting Exp 030 effnetv2b0_sydorskyi v2 (EffNetV2-B0 + Sydorskyi mixup, no dual_val) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/030-sydorskyi-mixup/checkpoints/2026-exp030-effnetv2b0-sydorskyi-v2")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP030_DIR}/configs/effnetv2b0_sydorskyi.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnetv2-b0-sed-sydorskyi/
}

# Exp 030 V3 — seed=43 diagnostic retrains. Same recipes as the v2 cells above;
# only data.seed differs (42 → 43). Run names have -seed43 suffix.

export_hgnet_b0_sed_v5j_control_seed43() {
    echo "=== Exporting Exp 030 v5j_control seed=43 (HGNetV2-B0, no dual_val) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/030-sydorskyi-mixup/checkpoints/2026-exp030-v5j-control-seed43")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP030_DIR}/configs/v5j_control_seed43.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v5j-control-seed43/
}

export_hgnet_b0_sed_v5j_sydorskyi_seed43() {
    echo "=== Exporting Exp 030 v5j_sydorskyi seed=43 (HGNetV2-B0 + Sydorskyi wave-mixup) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/030-sydorskyi-mixup/checkpoints/2026-exp030-v5j-sydorskyi-seed43")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP030_DIR}/configs/v5j_sydorskyi_seed43.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v5j-sydorskyi-seed43/
}

export_effnetv2_b0_sed_control_seed43() {
    echo "=== Exporting Exp 030 effnetv2b0_control seed=43 (EfficientNetV2-B0) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/030-sydorskyi-mixup/checkpoints/2026-exp030-effnetv2b0-control-seed43")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP030_DIR}/configs/effnetv2b0_control_seed43.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnetv2-b0-sed-control-seed43/
}

export_effnetv2_b0_sed_sydorskyi_seed43() {
    echo "=== Exporting Exp 030 effnetv2b0_sydorskyi seed=43 (EffNetV2-B0 + Sydorskyi mixup) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/030-sydorskyi-mixup/checkpoints/2026-exp030-effnetv2b0-sydorskyi-seed43")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP030_DIR}/configs/effnetv2b0_sydorskyi_seed43.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnetv2-b0-sed-sydorskyi-seed43/
}

# ============================================================================
# Exp 031: Perch v2 knowledge distillation on v5j (Tucker recipe, Phase 1)
# ============================================================================

export_hgnet_b0_sed_v5j_kd() {
    # Phase 1: v5j with Perch v2 embedding-space KD branch.
    # Distill head is excluded from the ONNX export (export script builds
    # SEDFinetuneModule without use_perch_distill, so distill_head.* keys
    # land in `unexpected` under strict=False and are ignored).
    echo "=== Exporting Exp 031 v5j_kd (HGNetV2-B0 + Perch v2 KD) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/031-perch-kd-distillation/checkpoints/2026-exp031-v5j-kd")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP031_DIR}/configs/v5j_kd.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v5j-kd/
}

export_hgnet_b0_sed_v5j_kd_last() {
    # Best-vs-last diagnostic: v5j_kd ep 19 weights from the save_top_k=-1
    # rerun. Tests whether the 0.890 LB on best-by-val ep 17 was a
    # ckpt-selection artifact vs structural KD failure.
    echo "=== Exporting Exp 031 v5j_kd LAST (ep 19) — best-vs-last diagnostic ==="
    local CKPT="${CHECKPOINTS_BASE}/031-perch-kd-distillation/checkpoints/2026-exp031-v5j-kd-resave/best-epochepoch=19.ckpt"
    if [ ! -f "$CKPT" ]; then
        echo "ERROR: ep 19 checkpoint not found at $CKPT" >&2
        echo "Has kd_resave finished training? Check squeue and the checkpoint dir." >&2
        exit 1
    fi
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP031_DIR}/configs/v5j_kd_resave.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v5j-kd-last/
}

export_hgnet_b0_sed_v5j_eracode() {
    # v5j-era codebase regression diagnostic: retrained v5j_control using the
    # v5j-era source code (worktree at commit e420633) in today's environment.
    # Offline metrics match v5j-original exactly (val/cmap 0.6429 = original
    # 0.6429); this submission confirms whether ep 19 also recovers v5j-original
    # LB 0.906 — proving the regression is in the HEAD codebase, not env/HW.
    # Same checkpoint dir naming as v5j-resubmit (017-mel-overhaul).
    echo "=== Exporting v5j-eracode (May 10 retrain w/ v5j-era code, drift diag) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-exp019-v5j-diag-eracode-may10")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP019_DIR}/config/hgnet_b0_sed_v5j.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v5j-eracode/
}

export_hgnet_b0_sed_v5j_control_last() {
    # Best-vs-last diagnostic: v5j_control ep 19 weights from the
    # save_top_k=-1 rerun. Tests whether ep 19 weights recover v5j-original
    # 0.906 LB on the new codebase, isolating ckpt-selection sensitivity
    # from real codebase drift.
    echo "=== Exporting Exp 031 v5j_control LAST (ep 19) — best-vs-last diagnostic ==="
    local CKPT="${CHECKPOINTS_BASE}/031-perch-kd-distillation/checkpoints/2026-exp031-v5j-control-resave/best-epochepoch=19.ckpt"
    if [ ! -f "$CKPT" ]; then
        echo "ERROR: ep 19 checkpoint not found at $CKPT" >&2
        exit 1
    fi
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP031_DIR}/configs/v5j_control_resave.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v5j-control-last/
}

export_hgnet_b0_sed_v5j_fmax() {
    echo "=== Exporting HGNetV2-B0 SED v5j-fmax (framewise-max only, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5j")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP019_DIR}/config/hgnet_b0_sed_v5j.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v5j-fmax/ \
        --blend-clip 0.0 --blend-frame 1.0
    cp artifacts/weights/species_mapping_2026_234.json artifacts/weights/hgnet-b0-sed-v5j-fmax/species_mapping.json
}

export_hgnet_b0_sed_v5j_clip() {
    echo "=== Exporting HGNetV2-B0 SED v5j-clip (clipwise-only / LSE, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5j")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP019_DIR}/config/hgnet_b0_sed_v5j.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v5j-clip/ \
        --blend-clip 1.0 --blend-frame 0.0
    cp artifacts/weights/species_mapping_2026_234.json artifacts/weights/hgnet-b0-sed-v5j-clip/species_mapping.json
}

export_hgnet_b0_sed_v5g() {
    echo "=== Exporting HGNetV2-B0 SED v5g (ttahara full reproduction, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5g")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP019_DIR}/config/hgnet_b0_sed_v5g.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v5g/
}

export_hgnet_b0_sed_v5g_4fold() {
    echo "=== Exporting HGNetV2-B0 SED v5g 4-fold ensemble (234 classes) ==="
    local OUTPUT_DIR="artifacts/weights/hgnet-b0-sed-v5g-4fold"
    mkdir -p "$OUTPUT_DIR"

    for FOLD in 0 1 2 3; do
        local CKPT_DIR="${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5g/fold${FOLD}"
        local CKPT
        CKPT="$(_find_best_ckpt "$CKPT_DIR")"
        echo ""
        echo "--- Fold ${FOLD}: $CKPT ---"
        python scripts/export_efficientnet_sed.py \
            --checkpoint "$CKPT" \
            --config "${EXP019_DIR}/config/hgnet_b0_sed_v5g.yaml" \
            --train-csv "$TAXONOMY_CSV_2026" \
            --output-dir "${OUTPUT_DIR}/fold${FOLD}/"
    done

    # Copy shared artifacts to top-level and rename per-fold ONNX models
    cp "${OUTPUT_DIR}/fold0/species_list.json" "${OUTPUT_DIR}/"
    cp "${OUTPUT_DIR}/fold0/mel_config.json" "${OUTPUT_DIR}/"
    # species_mapping.json is built by build_species_mapping.py, not by export.
    # Copy from single-model export if available, or from the mapping build step.
    local SINGLE_MAPPING="artifacts/weights/hgnet-b0-sed-v5g/species_mapping.json"
    local MAPPING_2026="artifacts/weights/species_mapping_2026.json"
    if [ -f "$SINGLE_MAPPING" ]; then
        cp "$SINGLE_MAPPING" "${OUTPUT_DIR}/"
    elif [ -f "$MAPPING_2026" ]; then
        cp "$MAPPING_2026" "${OUTPUT_DIR}/species_mapping.json"
    else
        echo "WARNING: species_mapping.json not found. Run single-model export first or build_species_mapping.py."
    fi
    for FOLD in 0 1 2 3; do
        cp "${OUTPUT_DIR}/fold${FOLD}/model.onnx" "${OUTPUT_DIR}/model_fold${FOLD}.onnx"
    done
    echo ""
    echo "4-fold ensemble exported to ${OUTPUT_DIR}/"
    ls -lh "${OUTPUT_DIR}"/model_fold*.onnx
}

export_hgnet_b0_sed_v5j_4fold() {
    echo "=== Exporting HGNetV2-B0 SED v5j 4-fold ensemble (234 classes) ==="
    local OUTPUT_DIR="artifacts/weights/hgnet-b0-sed-v5j-4fold"
    mkdir -p "$OUTPUT_DIR"

    for FOLD in 0 1 2 3; do
        local CKPT_DIR="${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5j/fold${FOLD}"
        local CKPT
        CKPT="$(_find_best_ckpt "$CKPT_DIR")"
        echo ""
        echo "--- Fold ${FOLD}: $CKPT ---"
        python scripts/export_efficientnet_sed.py \
            --checkpoint "$CKPT" \
            --config "${EXP019_DIR}/config/hgnet_b0_sed_v5j.yaml" \
            --train-csv "$TAXONOMY_CSV_2026" \
            --output-dir "${OUTPUT_DIR}/fold${FOLD}/"
    done

    # Copy shared artifacts to top-level and rename per-fold ONNX models
    cp "${OUTPUT_DIR}/fold0/species_list.json" "${OUTPUT_DIR}/"
    cp "${OUTPUT_DIR}/fold0/mel_config.json" "${OUTPUT_DIR}/"
    local SINGLE_MAPPING="artifacts/weights/hgnet-b0-sed-v5j/species_mapping.json"
    local MAPPING_2026="artifacts/weights/species_mapping_2026.json"
    if [ -f "$SINGLE_MAPPING" ]; then
        cp "$SINGLE_MAPPING" "${OUTPUT_DIR}/"
    elif [ -f "$MAPPING_2026" ]; then
        cp "$MAPPING_2026" "${OUTPUT_DIR}/species_mapping.json"
    else
        echo "WARNING: species_mapping.json not found. Run single-model export first or build_species_mapping.py."
    fi
    for FOLD in 0 1 2 3; do
        cp "${OUTPUT_DIR}/fold${FOLD}/model.onnx" "${OUTPUT_DIR}/model_fold${FOLD}.onnx"
    done
    echo ""
    echo "4-fold ensemble exported to ${OUTPUT_DIR}/"
    ls -lh "${OUTPUT_DIR}"/model_fold*.onnx
}

# ============================================================================
# Exp 021: SED training Phase C (AttHead + backbone scaling)
# ============================================================================

EXP021_DIR="$(dirname "$EXPERIMENT_DIR")/021-sed-training"
EXP026_DIR="$(dirname "$EXPERIMENT_DIR")/026-domain-shift-training"
EXP022_DIR="$(dirname "$EXPERIMENT_DIR")/022-pcen-normalization"
EXP025_DIR="$(dirname "$EXPERIMENT_DIR")/025-iter-pseudo-labeling"
EXP027_DIR="$(dirname "$EXPERIMENT_DIR")/027-model-training-fixes"
EXP029_DIR="$(dirname "$EXPERIMENT_DIR")/029-2025-pretrain"
EXP030_DIR="$(dirname "$EXPERIMENT_DIR")/030-sydorskyi-mixup"
EXP031_DIR="$(dirname "$EXPERIMENT_DIR")/031-perch-kd-distillation"
EXP032_DIR="$(dirname "$EXPERIMENT_DIR")/032-tucker-distilled-sed"
EXP033_DIR="$(dirname "$EXPERIMENT_DIR")/033-own-teacher-pl"
EXP037_DIR="$(dirname "$EXPERIMENT_DIR")/037-noisy-student-pl"

export_hgnet_b2_sed_v6h() {
    echo "=== Exporting HGNetV2-B2 SED v6h (AttHead + BCE, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/021-sed-training/checkpoints/2026-hgnet-b2-sed-v6h")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP021_DIR}/config/hgnet_b2_sed_v6h.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b2-sed-v6h/ \
        --blend-clip 0.75 --blend-frame 0.25
}

export_hgnet_b0_sed_v6b() {
    echo "=== Exporting HGNetV2-B0 SED v6b (waveform MixUp, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/021-sed-training/checkpoints/2026-hgnet-b0-sed-v6b")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP021_DIR}/config/hgnet_b0_sed_v6b.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v6b/
}

export_effnet_b0_ns_sed_v6j() {
    echo "=== Exporting EffNet-B0 NS SED v6j (AttHead + BCE, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/021-sed-training/checkpoints/2026-effnet-b0-ns-sed-v6j")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP021_DIR}/config/effnet_b0_ns_sed_v6j.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-ns-sed-v6j/ \
        --blend-clip 0.75 --blend-frame 0.25
}

# Exp 021 Phase E: LSEHead multi-context
export_hgnet_b0_mc_lse_sed_v8a() {
    echo "=== Exporting HGNetV2-B0 MC-LSE SED v8a (3×5s multi-context, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/021-sed-training/checkpoints/2026-hgnet-b0-mc-lse-sed-v8a")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP021_DIR}/config/hgnet_b0_mc_lse_sed_v8a.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-mc-lse-sed-v8a/
}

export_hgnet_b0_mc_lse_sed_v8b() {
    echo "=== Exporting HGNetV2-B0 MC-LSE SED v8b (4×5s multi-context, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/021-sed-training/checkpoints/2026-hgnet-b0-mc-lse-sed-v8b")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP021_DIR}/config/hgnet_b0_mc_lse_sed_v8b.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-mc-lse-sed-v8b/
}

export_hgnet_b0_atthead_v6k() {
    echo "=== Exporting HGNetV2-B0 AttHead v6k (competition-standard AttHead, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/021-sed-training/checkpoints/2026-hgnet-b0-atthead-v6k")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP021_DIR}/config/hgnet_b0_atthead_v6k.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-atthead-v6k/ \
        --blend-clip 0.0 --blend-frame 1.0
    cp artifacts/weights/species_mapping_2026_234.json artifacts/weights/hgnet-b0-atthead-v6k/species_mapping.json
}

# Exp 022: PCEN normalization
export_hgnet_b0_sed_v10a() {
    echo "=== Exporting HGNetV2-B0 SED v10a (learnable PCEN, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/022-pcen-normalization/checkpoints/2026-hgnet-b0-sed-v10a")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP022_DIR}/config/hgnet_b0_sed_v10a.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v10a/
}

export_hgnet_b0_sed_v10b() {
    echo "=== Exporting HGNetV2-B0 SED v10b (fixed PCEN, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/022-pcen-normalization/checkpoints/2026-hgnet-b0-sed-v10b")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP022_DIR}/config/hgnet_b0_sed_v10b.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v10b/
}

export_hgnet_b0_sed_v10c() {
    echo "=== Exporting HGNetV2-B0 SED v10c (PCEN+AmplitudeToDB, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/022-pcen-normalization/checkpoints/2026-hgnet-b0-sed-v10c")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP022_DIR}/config/hgnet_b0_sed_v10c.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v10c/
}

export_hgnet_b0_sed_v5j_repro() {
    echo "=== Exporting HGNetV2-B0 SED v5j-repro (v5j reproduction from exp 022, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/022-pcen-normalization/checkpoints/2026-hgnet-b0-sed-v5j-repro")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP022_DIR}/config/hgnet_b0_sed_v5j.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v5j-repro/
}

export_hgnet_b0_sed_v5j_dualval() {
    echo "=== Exporting HGNetV2-B0 SED v5j-dualval (dual val checkpoint, exp 026, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/026-domain-shift-training/checkpoints/2026-exp026-v5j-dualval")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP019_DIR}/config/hgnet_b0_sed_v5j.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v5j-dualval/
}

export_hgnet_b0_sed_v9a_10s_att() {
    echo "=== Exporting HGNetV2-B0 SED v9a (10s + AttHead, exp 026, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/026-domain-shift-training/checkpoints/2026-exp026-v9a-10s-att")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP026_DIR}/configs/v9a_10s_att.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v9a-10s-att/ \
        --blend-clip 0.0 --blend-frame 1.0
}

export_hgnet_b0_sed_v9b_15s_att() {
    echo "=== Exporting HGNetV2-B0 SED v9b (15s + AttHead, exp 026, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/026-domain-shift-training/checkpoints/2026-exp026-v9b-15s-att")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP026_DIR}/configs/v9b_15s_att.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v9b-15s-att/ \
        --blend-clip 0.0 --blend-frame 1.0
}

export_hgnet_b0_sed_v9c_20s_att() {
    echo "=== Exporting HGNetV2-B0 SED v9c (20s + AttHead, exp 026, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/026-domain-shift-training/checkpoints/2026-exp026-v9c-20s-att")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP026_DIR}/configs/v9c_20s_att.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v9c-20s-att/ \
        --blend-clip 0.0 --blend-frame 1.0
}

export_hgnet_b0_sed_v5j_pl_v1() {
    echo "=== Exporting HGNetV2-B0 SED v5j-pl-v1 (pseudo-label trained, exp 025, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/025-iter-pseudo-labeling/checkpoints/2026-exp025-v1-pl-round1")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP019_DIR}/config/hgnet_b0_sed_v5j.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v5j-pl-v1/
}

export_hgnet_b0_sed_v3_waveform_mix() {
    echo "=== Exporting HGNetV2-B0 SED v3-waveform-mix (PL waveform mixing, exp 025 Phase C, 234 classes) ==="
    local CKPT="${CHECKPOINTS_BASE}/025-iter-pseudo-labeling/checkpoints/2026-exp025-v3-waveform-mix/best-epochepoch=38.ckpt"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP019_DIR}/config/hgnet_b0_sed_v5j.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v3-waveform-mix/
}

export_hgnet_b0_sed_v6l() {
    echo "=== Exporting HGNetV2-B0 SED v6l (manifold mixup, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/021-sed-training/checkpoints/2026-hgnet-b0-sed-v6l")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP021_DIR}/config/hgnet_b0_sed_v6l.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v6l/
}

# ============================================================================
# Exp 027: Model training fixes (additive MixUp + CE loss)
# ============================================================================

export_hgnet_b0_sed_v10a_exp027() {
    echo "=== Exporting HGNetV2-B0 SED v10a (additive MixUp + CE loss, exp 027, 234 classes) ==="
    local CKPT="${CHECKPOINTS_BASE}/027-model-training-fixes/checkpoints/2026-exp027-v10a-addmix-ce/last.ckpt"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP027_DIR}/configs/v10a_addmix_ce.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v10a-exp027/
}

export_hgnet_b0_sed_v11b_exp027() {
    echo "=== Exporting HGNetV2-B0 SED v11b (simple mel norm, exp 027, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/027-model-training-fixes/checkpoints/2026-exp027-v11b-melnorm")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP027_DIR}/configs/v11b_melnorm.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v11b-exp027/
}

export_hgnet_b0_sed_v10b_exp027() {
    echo "=== Exporting HGNetV2-B0 SED v10b (additive MixUp + BCE, exp 027, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/027-model-training-fixes/checkpoints/2026-exp027-v10b-addmix-bce")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP027_DIR}/configs/v10b_addmix_bce.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/hgnet-b0-sed-v10b-exp027/
}

# ============================================================================
# Exp 032: Tucker distilled-SED full replication (paired control / KD)
# Uses dedicated export_distilled_sed.py (AttHeadLogitSpace + logit-space output).
# Submits last.ckpt (epoch 24) per the "submit last" decision after the offline
# ranker on val fold 0 showed noisy results (n_evaluable=29 → small variance).
# ============================================================================

export_distilled_sed_control() {
    # Auto-pick the most recent last*.ckpt. Phase 1 produced last.ckpt + last-v1.ckpt
    # (broken NNAudio mel); Phase D retrain produced last-v2.ckpt (Tucker mel fix).
    # `find_last_ckpt` returns the newest by mtime so future retrains land cleanly.
    echo "=== Exporting Exp 032 distilled-SED CONTROL (Tucker recipe, KD off) ==="
    local CKPT
    CKPT="$(find_last_ckpt "${CHECKPOINTS_BASE}/032-tucker-distilled-sed/checkpoints/2026-exp032-distilled-sed-control")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_distilled_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP032_DIR}/configs/distilled_sed_control.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/distilled-sed-control/
}

export_distilled_sed_kd() {
    # Same auto-pick rationale as control.
    echo "=== Exporting Exp 032 distilled-SED KD (Tucker recipe + Perch v2 KD) ==="
    local CKPT
    CKPT="$(find_last_ckpt "${CHECKPOINTS_BASE}/032-tucker-distilled-sed/checkpoints/2026-exp032-distilled-sed-kd")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_distilled_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP032_DIR}/configs/distilled_sed_kd.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/distilled-sed-kd/
}

export_distilled_sed_kd_5fold() {
    # 5-fold ensemble — fold 0 uses the LB-validated last-v2.ckpt at the
    # top level (from the Phase D retrain that scored LB 0.906); folds 1-4
    # use fold{N}/last*.ckpt (newest mtime) from --fold N submissions.
    # Tucker inference: per-fold blended logits -> mean across folds in
    # LOGIT space -> 5-tap Gauss smooth -> single sigmoid (audit sec 7.3).
    echo "=== Exporting Exp 032 distilled-SED KD 5-fold ensemble ==="
    local OUTPUT_DIR="artifacts/weights/distilled-sed-kd-5fold"
    local BASE_CKPT_DIR="${CHECKPOINTS_BASE}/032-tucker-distilled-sed/checkpoints/2026-exp032-distilled-sed-kd"
    mkdir -p "$OUTPUT_DIR"

    for FOLD in 0 1 2 3 4; do
        local CKPT
        if [ "$FOLD" = "0" ]; then
            # Fold 0: LB-validated last-v2.ckpt at top level
            CKPT="$(find_last_ckpt "$BASE_CKPT_DIR")"
        else
            local FOLD_DIR="$BASE_CKPT_DIR/fold${FOLD}"
            if [ ! -d "$FOLD_DIR" ]; then
                echo "ERROR: fold ${FOLD} dir not found at $FOLD_DIR" >&2
                echo "Did you submit `bash sbatch/run.sh configs/distilled_sed_kd.yaml --fold ${FOLD}`?" >&2
                exit 1
            fi
            CKPT="$(find_last_ckpt "$FOLD_DIR")"
        fi
        echo ""
        echo "--- Fold ${FOLD}: $CKPT ---"
        python scripts/export_distilled_sed.py \
            --checkpoint "$CKPT" \
            --config "${EXP032_DIR}/configs/distilled_sed_kd.yaml" \
            --train-csv "$TAXONOMY_CSV_2026" \
            --output-dir "${OUTPUT_DIR}/fold${FOLD}/"
    done

    # Shared artifacts at the top level (species_list + mel_config from fold 0;
    # species_mapping from the 234-class identity map exp 030/031 use).
    cp "${OUTPUT_DIR}/fold0/species_list.json" "${OUTPUT_DIR}/"
    cp "${OUTPUT_DIR}/fold0/mel_config.json" "${OUTPUT_DIR}/"
    local MAPPING_2026="artifacts/weights/species_mapping_2026_234.json"
    if [ -f "$MAPPING_2026" ]; then
        cp "$MAPPING_2026" "${OUTPUT_DIR}/species_mapping.json"
    elif [ -f "artifacts/weights/distilled-sed-kd/species_mapping.json" ]; then
        cp "artifacts/weights/distilled-sed-kd/species_mapping.json" "${OUTPUT_DIR}/species_mapping.json"
    fi

    # Per-fold ONNX at the top level (the kernel detects ensemble mode via
    # `model_fold*.onnx` glob)
    for FOLD in 0 1 2 3 4; do
        cp "${OUTPUT_DIR}/fold${FOLD}/model.onnx" "${OUTPUT_DIR}/model_fold${FOLD}.onnx"
    done

    echo ""
    echo "5-fold ensemble exported to ${OUTPUT_DIR}/"
    ls -lh "${OUTPUT_DIR}"/model_fold*.onnx
}

export_distilled_sed_kd_hgnet() {
    # Phase 2.D ablation: distilled-SED KD with HGNetV2-B0 backbone
    # (v5j's choice) instead of EffNet-B0 NS-JFT. Tests whether the
    # +0.041 control-vs-v5j gap was driven by backbone choice. Same
    # recipe otherwise: AttHeadLogitSpace + dual-supervision + Perch v2
    # KD + TuckerMel + waveform mixup + SpecAugment + 25ep cosine.
    echo "=== Exporting Exp 032 distilled-SED KD HGNet variant (Phase 2.D ablation) ==="
    local CKPT
    CKPT="$(find_last_ckpt "${CHECKPOINTS_BASE}/032-tucker-distilled-sed/checkpoints/2026-exp032-distilled-sed-kd-hgnet")"
    echo "Using checkpoint: $CKPT"
    local OUTPUT_DIR="artifacts/weights/distilled-sed-kd-hgnet"
    python scripts/export_distilled_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP032_DIR}/configs/distilled_sed_kd_hgnet.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir "${OUTPUT_DIR}/"

    # Copy the 234-class identity species_mapping (kernel needs it for the
    # output column ordering). Same file used by all distilled-sed-kd variants.
    local MAPPING_2026="artifacts/weights/species_mapping_2026_234.json"
    if [ -f "$MAPPING_2026" ]; then
        cp "$MAPPING_2026" "${OUTPUT_DIR}/species_mapping.json"
    elif [ -f "artifacts/weights/distilled-sed-kd/species_mapping.json" ]; then
        cp "artifacts/weights/distilled-sed-kd/species_mapping.json" "${OUTPUT_DIR}/species_mapping.json"
    fi
}

export_distilled_sed_kd_single_fold() {
    # Phase 2.A diagnostic: export a single fold of the 5-fold KD ensemble
    # for standalone LB submission. Characterizes per-fold LB variance and
    # the offline↔LB correlation across folds (offline val/auroc range
    # 0.908-0.931 → expected LB band ~0.88-0.93 at single-fold).
    # Fold 0 is already exported as `distilled-sed-kd` (Phase D, LB 0.906);
    # this helper is for folds 1-4.
    local FOLD="$1"
    if [ -z "$FOLD" ] || ! [[ "$FOLD" =~ ^[1-4]$ ]]; then
        echo "ERROR: export_distilled_sed_kd_single_fold requires fold in [1, 4], got '${FOLD}'" >&2
        exit 1
    fi
    echo "=== Exporting Exp 032 distilled-SED KD fold ${FOLD} (Phase 2.A single-fold diagnostic) ==="
    local CKPT
    CKPT="$(find_last_ckpt "${CHECKPOINTS_BASE}/032-tucker-distilled-sed/checkpoints/2026-exp032-distilled-sed-kd/fold${FOLD}")"
    echo "Using checkpoint: $CKPT"
    local OUTPUT_DIR="artifacts/weights/distilled-sed-kd-fold${FOLD}"
    python scripts/export_distilled_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP032_DIR}/configs/distilled_sed_kd.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir "${OUTPUT_DIR}/"

    # Copy the 234-class identity species_mapping (kernel needs it).
    local MAPPING_2026="artifacts/weights/species_mapping_2026_234.json"
    if [ -f "$MAPPING_2026" ]; then
        cp "$MAPPING_2026" "${OUTPUT_DIR}/species_mapping.json"
    elif [ -f "artifacts/weights/distilled-sed-kd/species_mapping.json" ]; then
        cp "artifacts/weights/distilled-sed-kd/species_mapping.json" "${OUTPUT_DIR}/species_mapping.json"
    fi
}

# ============================================================================
# Exp 033 own-teacher PL ablations — 10 cells of distilled-sed-kd recipe +
# 1:1 Antoine-style pseudo-labels. Same Tucker recipe + KD + AttHeadLogitSpace
# as exp 032's distilled-sed-kd; the only differences vs exp 032 are training-
# time (PL dataloader, sampler, loss_weight=1.0). Inference path is identical,
# so we reuse export_distilled_sed.py.
#
# T1 (`*-v5j`) ablations: ckpt dir holds `last.ckpt` (stale, from cancelled
# num_workers=0 first wave) AND `last-v1.ckpt` (the real final from the
# relaunched run). `find_last_ckpt` picks newest mtime → last-v1.ckpt.
# T2 (`*-kd`) ablations: single clean launch → just `last.ckpt`.
# ============================================================================

export_distilled_sed_pl() {
    # Generic helper: export one exp 033 ablation cell.
    # Args: $1 = ablation id (a1..a5), $2 = teacher (v5j|kd), $3 = optional version (v3, v4, ...)
    # When $3 is provided, slug/config/ckpt-dir all get a trailing -v3 / _v3 marker
    # (e.g. distilled-sed-pl-a2-kd-v3, configs/a2_kd_v3.yaml, 2026-exp033-a2-kd-v3/fold0).
    local AB="$1"
    local TEACHER="$2"
    local VERSION="${3:-}"
    if ! [[ "$AB" =~ ^a[1-5]$ ]] || ! [[ "$TEACHER" =~ ^(v5j|kd)$ ]]; then
        echo "ERROR: export_distilled_sed_pl: invalid args (ablation='$AB', teacher='$TEACHER')" >&2
        exit 1
    fi
    if [[ -n "$VERSION" ]] && ! [[ "$VERSION" =~ ^v[3-9]$ ]]; then
        echo "ERROR: export_distilled_sed_pl: invalid version '$VERSION' (expect v3..v9)" >&2
        exit 1
    fi
    local VDASH="${VERSION:+-${VERSION}}"
    local VUNDER="${VERSION:+_${VERSION}}"
    local SLUG="distilled-sed-pl-${AB}-${TEACHER}${VDASH}"
    local CKPT_RUN_DIR="${CHECKPOINTS_BASE}/033-own-teacher-pl/checkpoints/2026-exp033-${AB}-${TEACHER}${VDASH}/fold0"
    echo "=== Exporting Exp 033 ${SLUG} (own-teacher PL on distilled-sed-kd) ==="
    local CKPT
    CKPT="$(find_last_ckpt "${CKPT_RUN_DIR}")"
    echo "Using checkpoint: $CKPT"
    local OUTPUT_DIR="artifacts/weights/${SLUG}"
    python scripts/export_distilled_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP033_DIR}/configs/${AB}_${TEACHER}${VUNDER}.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir "${OUTPUT_DIR}/"

    # 234-class identity species mapping (shared file from exp 030/031/032).
    local MAPPING_2026="artifacts/weights/species_mapping_2026_234.json"
    if [ -f "$MAPPING_2026" ]; then
        cp "$MAPPING_2026" "${OUTPUT_DIR}/species_mapping.json"
    elif [ -f "artifacts/weights/distilled-sed-kd/species_mapping.json" ]; then
        cp "artifacts/weights/distilled-sed-kd/species_mapping.json" "${OUTPUT_DIR}/species_mapping.json"
    fi
}

# ============================================================================
# Exp 037 Noisy-Student soft-PL round-1 LB picks. Same distilled-sed-kd recipe
# + inference path as exp 032/033 (EffNet-B0 + AttHeadLogitSpace + Tucker mel),
# so we reuse export_distilled_sed.py. The ONLY difference vs exp 033's helper:
# we export the BEST-BY-VAL/AUROC epoch checkpoint (explicit best-epoch*.ckpt),
# not find_last_ckpt — round 1 used save_top_k=-1 (all epochs saved) and the
# competition metric is val/auroc. Best epochs (fold 0) read from TensorBoard.
# ============================================================================
export_exp037_pl() {
    # Args: $1 = cell (mixed-case, matches run_name suffix + config basename),
    #       $2 = best-by-val/auroc epoch (zero-padded, e.g. 03 / 11 / 19).
    local CELL="$1"
    local EP="$2"
    local SLUG="exp037-$(echo "$CELL" | tr '[:upper:]' '[:lower:]')"
    local CFG="${CELL//-/_}"  # r1-A-raw-augF -> r1_A_raw_augF (config basename)
    local CKPT="${CHECKPOINTS_BASE}/037-noisy-student-pl/checkpoints/2026-exp037-${CELL}/fold0/best-epochepoch=${EP}.ckpt"
    echo "=== Exporting Exp 037 ${SLUG} (best val/auroc @ ep${EP}) ==="
    if [ ! -f "$CKPT" ]; then
        echo "ERROR: checkpoint not found: $CKPT" >&2
        exit 1
    fi
    echo "Using checkpoint: $CKPT"
    local OUTPUT_DIR="artifacts/weights/${SLUG}"
    python scripts/export_distilled_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP037_DIR}/configs/${CFG}.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir "${OUTPUT_DIR}/"

    # 234-class identity species mapping (built by the species-mapping case block).
    local MAPPING_2026="artifacts/weights/species_mapping_2026_234.json"
    if [ -f "$MAPPING_2026" ]; then
        cp "$MAPPING_2026" "${OUTPUT_DIR}/species_mapping.json"
    fi
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

# ===== Exp 034: Pretrained-backbone yoink (Sydorskyi EffNetV2-S), 234 classes =====
# Full-model audio→probs ONNX (nnAudio mel baked into the graph; see export_xcpretrain.py).
EXP034_DIR="$(dirname "$EXPERIMENT_DIR")/034-pretrained-backbone"
# ===== Exp 036: XC-backbone faithful fine-tune (same model as exp 034, gentler LR) =====
# Reuses export_xcpretrain.py (identical XCPretrainWaveClassifier graph; PP baked in).
EXP036_DIR="$(dirname "$EXPERIMENT_DIR")/036-xc-backbone-finetune"

export_xcpretrain_effv2s_frozen() {
    echo "=== Exporting XCPretrain EffNetV2-S frozen (exp 034, fold 0) ==="
    local CKPT="${CHECKPOINTS_BASE}/034-pretrained-backbone/checkpoints/xcpretrain-effv2s-frozen/fold0/last.ckpt"
    local OUT="artifacts/weights/xcpretrain-effv2s-frozen"
    echo "Using checkpoint: $CKPT"
    python scripts/export_xcpretrain.py \
        --checkpoint "$CKPT" \
        --config "${EXP034_DIR}/configs/xcpretrain-effv2s-frozen.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir "${OUT}/"
    cp artifacts/weights/species_mapping_2026_234.json "${OUT}/species_mapping.json"
}

export_xcpretrain_effv2s_fullft() {
    echo "=== Exporting XCPretrain EffNetV2-S full-FT (exp 034, fold 0) ==="
    local CKPT="${CHECKPOINTS_BASE}/034-pretrained-backbone/checkpoints/xcpretrain-effv2s-fullft/fold0/last.ckpt"
    local OUT="artifacts/weights/xcpretrain-effv2s-fullft"
    echo "Using checkpoint: $CKPT"
    python scripts/export_xcpretrain.py \
        --checkpoint "$CKPT" \
        --config "${EXP034_DIR}/configs/xcpretrain-effv2s-fullft.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir "${OUT}/"
    cp artifacts/weights/species_mapping_2026_234.json "${OUT}/species_mapping.json"
}

# Phase K: Perch v2 KD on the in-domain backbone (JOINT-gradient variant) — best
# offline val/auroc of the KD grid (0.9217 vs detach 0.8847). KD adds a DistillHead
# (`distill_head.*` in the Lightning ckpt) that is DISCARDED here: export_xcpretrain.py
# loads only the `model.*` subset, so the ONNX graph is identical to full-FT (KD
# changed weights, not architecture). PP (5-tap logit-space Gaussian smooth) is baked
# into mel_config.json by the export script → PP is ON for this kernel by default.
# last.ckpt = epoch 43 (top-3, val 0.9226); same last-top-k semantics as the
# LB-validated full-FT export (epoch 22).
export_xcpretrain_effv2s_kd_joint() {
    echo "=== Exporting XCPretrain EffNetV2-S KD joint (exp 034, fold 0) ==="
    local CKPT="${CHECKPOINTS_BASE}/034-pretrained-backbone/checkpoints/xcpretrain-effv2s-kd-joint/fold0/last.ckpt"
    local OUT="artifacts/weights/xcpretrain-effv2s-kd-joint"
    echo "Using checkpoint: $CKPT"
    python scripts/export_xcpretrain.py \
        --checkpoint "$CKPT" \
        --config "${EXP034_DIR}/configs/xcpretrain-effv2s-kd-joint.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir "${OUT}/"
    cp artifacts/weights/species_mapping_2026_234.json "${OUT}/species_mapping.json"
}

# Exp 036 Phase A winner: a1-lower-lr (lr 5e-5). Fixes exp 034 full-FT's overfitting
# collapse (peaks ep3 0.906 → decays 0.875); a1 climbs to a stable 0.91-0.92 plateau
# (ep19-50), best offline val/auroc 0.921. Same XCPretrainWaveClassifier graph as exp 034
# → export_xcpretrain.py applies UNCHANGED (model.* filter; PP baked into mel_config).
# Two checkpoint candidates LB-arbitrated (lead with SWA): swa-last5 (avg of ep45-49,
# the flat tail — selection-free, robust to the untrustworthy CV) and last.ckpt (ep49).
export_xcfinetune_effv2s_lowlr_swa() {
    echo "=== Exporting XCFinetune EffNetV2-S LowLR SWA (exp 036 a1, last-5 avg, fold 0) ==="
    local CKPT="${CHECKPOINTS_BASE}/036-xc-backbone-finetune/checkpoints/a1-lower-lr/fold0/swa-last5.ckpt"
    local OUT="artifacts/weights/xcfinetune-effv2s-lowlr-swa"
    echo "Using checkpoint: $CKPT"
    python scripts/export_xcpretrain.py \
        --checkpoint "$CKPT" \
        --config "${EXP036_DIR}/configs/a1-lower-lr.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir "${OUT}/"
    cp artifacts/weights/species_mapping_2026_234.json "${OUT}/species_mapping.json"
}

export_xcfinetune_effv2s_lowlr_last() {
    echo "=== Exporting XCFinetune EffNetV2-S LowLR last (exp 036 a1, ep49, fold 0) ==="
    local CKPT="${CHECKPOINTS_BASE}/036-xc-backbone-finetune/checkpoints/a1-lower-lr/fold0/last.ckpt"
    local OUT="artifacts/weights/xcfinetune-effv2s-lowlr-last"
    echo "Using checkpoint: $CKPT"
    python scripts/export_xcpretrain.py \
        --checkpoint "$CKPT" \
        --config "${EXP036_DIR}/configs/a1-lower-lr.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir "${OUT}/"
    cp artifacts/weights/species_mapping_2026_234.json "${OUT}/species_mapping.json"
}

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
    effnet-b0-sed-v1|effnet-b0-sed-v2|effnet-b0-sed-v3a|effnet-b0-sed-v3b|effnet-b0-sed-v3c|effnet-b0-sed-v3d|\
    effnet-b0-sed-v4a|effnet-b0-sed-v4b|effnet-b0-sed-v4b-fulldata|effnet-b0-sed-v4c|effnet-b0-sed-v4d|effnet-b0-sed-v4e|effnet-b0-sed-v4f|effnet-b0-sed-v4g|effnet-b0-sed-v4h|effnet-b0-sed-v4j|\
    effnet-b0-sed-v5a|effnet-b0-sed-v5b|effnet-b0-sed-v5c|effnet-b0-sed-v5d|hgnet-b0-sed-v5e|effnet-b0-sed-v5f|hgnet-b0-sed-v5g|hgnet-b0-sed-v5g-4fold|hgnet-b0-sed-v5h|hgnet-b0-sed-v5i|hgnet-b0-sed-v5j|hgnet-b0-sed-v5j-4fold|hgnet-b0-sed-v5k|hgnet-b0-sed-v5l|effnet-b0-sed-v5m|effnet-b0-sed-v5n|all-019-ablations|\
    hgnet-b0-sed-v5j-fmax|hgnet-b0-sed-v5j-clip|hgnet-b0-atthead-v6k|\
    hgnet-b0-sed-v6b|hgnet-b0-sed-v6l|hgnet-b2-sed-v6h|effnet-b0-ns-sed-v6j|hgnet-b0-mc-lse-sed-v8a|hgnet-b0-mc-lse-sed-v8b|\
    hgnet-b0-sed-v10a|hgnet-b0-sed-v10b|hgnet-b0-sed-v10c|hgnet-b0-sed-v5j-repro|\
    hgnet-b0-sed-v5j-pl-v1|hgnet-b0-sed-v3-waveform-mix|hgnet-b0-sed-v5j-dualval|\
    hgnet-b0-sed-v9a-10s-att|hgnet-b0-sed-v9b-15s-att|hgnet-b0-sed-v9c-20s-att|\
    hgnet-b0-sed-v10a-exp027|hgnet-b0-sed-v10b-exp027|hgnet-b0-sed-v11b-exp027|\
    hgnet-b0-sed-v5j-finetune-2026|hgnet-b0-sed-v5j-ft-splitlr|\
    hgnet-b0-sed-v5j-control|hgnet-b0-sed-v5j-sydorskyi|effnetv2-b0-sed-control|effnetv2-b0-sed-sydorskyi|\
    hgnet-b0-sed-v5j-control-seed43|hgnet-b0-sed-v5j-sydorskyi-seed43|effnetv2-b0-sed-control-seed43|effnetv2-b0-sed-sydorskyi-seed43|\
    hgnet-b0-sed-v5j-kd|hgnet-b0-sed-v5j-kd-last|hgnet-b0-sed-v5j-control-last|\
    hgnet-b0-sed-v5j-eracode|\
    hgnet-b0-sed-v5j-resubmit|\
    distilled-sed-control|distilled-sed-kd|all-032-distilled-sed|\
    distilled-sed-kd-5fold|distilled-sed-kd-hgnet|\
    distilled-sed-kd-fold1|distilled-sed-kd-fold2|distilled-sed-kd-fold3|distilled-sed-kd-fold4|\
    distilled-sed-pl-a1-v5j|distilled-sed-pl-a1-kd|\
    distilled-sed-pl-a2-v5j|distilled-sed-pl-a2-kd|\
    distilled-sed-pl-a3-v5j|distilled-sed-pl-a3-kd|\
    distilled-sed-pl-a4-v5j|distilled-sed-pl-a4-kd|\
    distilled-sed-pl-a5-v5j|distilled-sed-pl-a5-kd|\
    distilled-sed-pl-a2-kd-v3|\
    all-033-distilled-sed-pl-day1|all-033-distilled-sed-pl-day2|\
    exp037-r1-a-raw-augf|exp037-r1-b-raw-augf|exp037-r1-c-raw-augf|\
    exp037-r1-a-raw-augt-softauc|exp037-r1-c-raw-augt-softauc|all-037-pl|\
    xcpretrain-effv2s-frozen|xcpretrain-effv2s-fullft|xcpretrain-effv2s-kd-joint|all-034-xcpretrain|\
    xcfinetune-effv2s-lowlr-swa|xcfinetune-effv2s-lowlr-last|all-036-xcfinetune|\
    all-017-ablations|all-2026-sed)
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
    effnet-b0-sed-v2)          export_effnet_b0_sed_v2 ;;
    effnet-b0-sed-v3a)         export_effnet_b0_sed_v3a ;;
    effnet-b0-sed-v3b)         export_effnet_b0_sed_v3b ;;
    effnet-b0-sed-v3c)         export_effnet_b0_sed_v3c ;;
    effnet-b0-sed-v3d)         export_effnet_b0_sed_v3d ;;
    all-2026-sed)
        export_effnet_b0_sed_v1
        echo ""
        export_effnet_b0_sed_v2
        echo ""
        export_effnet_b0_sed_v3a
        echo ""
        export_effnet_b0_sed_v3b
        echo ""
        export_effnet_b0_sed_v3c
        echo ""
        export_effnet_b0_sed_v3d
        ;;
    effnet-b0-sed-v4a)         export_effnet_b0_sed_v4a ;;
    effnet-b0-sed-v4b)         export_effnet_b0_sed_v4b ;;
    effnet-b0-sed-v4c)         export_effnet_b0_sed_v4c ;;
    effnet-b0-sed-v4d)         export_effnet_b0_sed_v4d ;;
    effnet-b0-sed-v4e)         export_effnet_b0_sed_v4e ;;
    effnet-b0-sed-v4f)             export_effnet_b0_sed_v4f ;;
    effnet-b0-sed-v4b-fulldata)    export_effnet_b0_sed_v4b_fulldata ;;
    effnet-b0-sed-v4g)         export_effnet_b0_sed_v4g ;;
    effnet-b0-sed-v4h)         export_effnet_b0_sed_v4h ;;
    effnet-b0-sed-v4j)         export_effnet_b0_sed_v4j ;;
    effnet-b0-sed-v5a)         export_effnet_b0_sed_v5a ;;
    effnet-b0-sed-v5b)         export_effnet_b0_sed_v5b ;;
    effnet-b0-sed-v5c)         export_effnet_b0_sed_v5c ;;
    effnet-b0-sed-v5d)         export_effnet_b0_sed_v5d ;;
    hgnet-b0-sed-v5e)          export_hgnet_b0_sed_v5e ;;
    effnet-b0-sed-v5f)         export_effnet_b0_sed_v5f ;;
    effnet-b0-sed-v5m)         export_effnet_b0_sed_v5m ;;
    effnet-b0-sed-v5n)         export_effnet_b0_sed_v5n ;;
    hgnet-b0-sed-v5g)          export_hgnet_b0_sed_v5g ;;
    hgnet-b0-sed-v5g-4fold)    export_hgnet_b0_sed_v5g_4fold ;;
    hgnet-b0-sed-v5h)          export_hgnet_b0_sed_v5h ;;
    hgnet-b0-sed-v5k)          export_hgnet_b0_sed_v5k ;;
    hgnet-b0-sed-v5i)          export_hgnet_b0_sed_v5i ;;
    hgnet-b0-sed-v5j)          export_hgnet_b0_sed_v5j ;;
    hgnet-b0-sed-v5j-resubmit) export_hgnet_b0_sed_v5j_resubmit ;;
    hgnet-b0-sed-v5j-fmax)     export_hgnet_b0_sed_v5j_fmax ;;
    hgnet-b0-sed-v5j-clip)     export_hgnet_b0_sed_v5j_clip ;;
    hgnet-b0-sed-v5j-4fold)    export_hgnet_b0_sed_v5j_4fold ;;
    hgnet-b0-sed-v5j-finetune-2026) export_hgnet_b0_sed_v5j_finetune_2026 ;;
    hgnet-b0-sed-v5j-ft-splitlr) export_hgnet_b0_sed_v5j_ft_splitlr ;;
    hgnet-b0-sed-v5j-control)    export_hgnet_b0_sed_v5j_control ;;
    hgnet-b0-sed-v5j-sydorskyi)  export_hgnet_b0_sed_v5j_sydorskyi ;;
    effnetv2-b0-sed-control)     export_effnetv2_b0_sed_control ;;
    effnetv2-b0-sed-sydorskyi)   export_effnetv2_b0_sed_sydorskyi ;;
    hgnet-b0-sed-v5j-control-seed43)    export_hgnet_b0_sed_v5j_control_seed43 ;;
    hgnet-b0-sed-v5j-sydorskyi-seed43)  export_hgnet_b0_sed_v5j_sydorskyi_seed43 ;;
    effnetv2-b0-sed-control-seed43)     export_effnetv2_b0_sed_control_seed43 ;;
    effnetv2-b0-sed-sydorskyi-seed43)   export_effnetv2_b0_sed_sydorskyi_seed43 ;;
    hgnet-b0-sed-v5j-kd)                export_hgnet_b0_sed_v5j_kd ;;
    hgnet-b0-sed-v5j-kd-last)           export_hgnet_b0_sed_v5j_kd_last ;;
    hgnet-b0-sed-v5j-control-last)      export_hgnet_b0_sed_v5j_control_last ;;
    hgnet-b0-sed-v5j-eracode)           export_hgnet_b0_sed_v5j_eracode ;;
    distilled-sed-control)              export_distilled_sed_control ;;
    distilled-sed-kd)                   export_distilled_sed_kd ;;
    distilled-sed-kd-5fold)             export_distilled_sed_kd_5fold ;;
    distilled-sed-kd-hgnet)             export_distilled_sed_kd_hgnet ;;
    distilled-sed-kd-fold1)             export_distilled_sed_kd_single_fold 1 ;;
    distilled-sed-kd-fold2)             export_distilled_sed_kd_single_fold 2 ;;
    distilled-sed-kd-fold3)             export_distilled_sed_kd_single_fold 3 ;;
    distilled-sed-kd-fold4)             export_distilled_sed_kd_single_fold 4 ;;
    all-032-distilled-sed)
        export_distilled_sed_control
        echo ""
        export_distilled_sed_kd
        ;;
    # Exp 033 own-teacher PL ablations (10 cells = 5 protocols × 2 teachers)
    distilled-sed-pl-a1-v5j)            export_distilled_sed_pl a1 v5j ;;
    distilled-sed-pl-a1-kd)             export_distilled_sed_pl a1 kd ;;
    distilled-sed-pl-a2-v5j)            export_distilled_sed_pl a2 v5j ;;
    distilled-sed-pl-a2-kd)             export_distilled_sed_pl a2 kd ;;
    distilled-sed-pl-a3-v5j)            export_distilled_sed_pl a3 v5j ;;
    distilled-sed-pl-a3-kd)             export_distilled_sed_pl a3 kd ;;
    distilled-sed-pl-a4-v5j)            export_distilled_sed_pl a4 v5j ;;
    distilled-sed-pl-a4-kd)             export_distilled_sed_pl a4 kd ;;
    distilled-sed-pl-a5-v5j)            export_distilled_sed_pl a5 v5j ;;
    distilled-sed-pl-a5-kd)             export_distilled_sed_pl a5 kd ;;
    # Day-1 / Day-2 batched dispatch (top-5 by D.5 e2e val/auroc then bottom-5).
    all-033-distilled-sed-pl-day1)
        # Top-5 by D.5 e2e val/auroc (2026-05-27).
        export_distilled_sed_pl a4 kd  ; echo ""
        export_distilled_sed_pl a3 kd  ; echo ""
        export_distilled_sed_pl a2 kd  ; echo ""
        export_distilled_sed_pl a4 v5j ; echo ""
        export_distilled_sed_pl a3 v5j
        ;;
    all-033-distilled-sed-pl-day2)
        # Bottom-5 by D.5 e2e val/auroc.
        export_distilled_sed_pl a1 kd  ; echo ""
        export_distilled_sed_pl a2 v5j ; echo ""
        export_distilled_sed_pl a1 v5j ; echo ""
        export_distilled_sed_pl a5 kd  ; echo ""
        export_distilled_sed_pl a5 v5j
        ;;
    # Exp 033 Phase 5 Sydorskyi-divergence diagnostic — Test 1 (chunk discard + trim).
    # Same canonical Antoine cell (random init + Uniform sampler + T2 teacher + 25 ep)
    # as distilled-sed-pl-a2-kd, but pointed at the v3 PL parquet that has Sydorskyi's
    # min_max_prob=0.5 chunk discard + trim_min_prob=0.1 per-class trim applied.
    distilled-sed-pl-a2-kd-v3)          export_distilled_sed_pl a2 kd v3 ;;
    # Exp 037 Noisy-Student soft-PL round-1 LB picks (best-by-val/auroc epoch, fold 0).
    exp037-r1-a-raw-augf)               export_exp037_pl r1-A-raw-augF 19 ;;
    exp037-r1-b-raw-augf)               export_exp037_pl r1-B-raw-augF 11 ;;
    exp037-r1-c-raw-augf)               export_exp037_pl r1-C-raw-augF 11 ;;
    exp037-r1-a-raw-augt-softauc)       export_exp037_pl r1-A-raw-augT-softauc 12 ;;
    exp037-r1-c-raw-augt-softauc)       export_exp037_pl r1-C-raw-augT-softauc 03 ;;
    all-037-pl)
        export_exp037_pl r1-A-raw-augF 19 ; echo ""
        export_exp037_pl r1-B-raw-augF 11 ; echo ""
        export_exp037_pl r1-C-raw-augF 11 ; echo ""
        export_exp037_pl r1-A-raw-augT-softauc 12 ; echo ""
        export_exp037_pl r1-C-raw-augT-softauc 03
        ;;
    xcpretrain-effv2s-frozen)           export_xcpretrain_effv2s_frozen ;;
    xcpretrain-effv2s-fullft)           export_xcpretrain_effv2s_fullft ;;
    xcpretrain-effv2s-kd-joint)         export_xcpretrain_effv2s_kd_joint ;;
    all-034-xcpretrain)
        export_xcpretrain_effv2s_frozen ; echo ""
        export_xcpretrain_effv2s_fullft
        ;;
    xcfinetune-effv2s-lowlr-swa)        export_xcfinetune_effv2s_lowlr_swa ;;
    xcfinetune-effv2s-lowlr-last)       export_xcfinetune_effv2s_lowlr_last ;;
    all-036-xcfinetune)
        export_xcfinetune_effv2s_lowlr_swa ; echo ""
        export_xcfinetune_effv2s_lowlr_last
        ;;
    hgnet-b0-sed-v5l)          export_hgnet_b0_sed_v5l ;;
    hgnet-b0-sed-v6b)          export_hgnet_b0_sed_v6b ;;
    hgnet-b2-sed-v6h)          export_hgnet_b2_sed_v6h ;;
    effnet-b0-ns-sed-v6j)      export_effnet_b0_ns_sed_v6j ;;
    hgnet-b0-mc-lse-sed-v8a)   export_hgnet_b0_mc_lse_sed_v8a ;;
    hgnet-b0-mc-lse-sed-v8b)   export_hgnet_b0_mc_lse_sed_v8b ;;
    hgnet-b0-atthead-v6k)      export_hgnet_b0_atthead_v6k ;;
    hgnet-b0-sed-v6l)          export_hgnet_b0_sed_v6l ;;
    hgnet-b0-sed-v10a)         export_hgnet_b0_sed_v10a ;;
    hgnet-b0-sed-v10b)         export_hgnet_b0_sed_v10b ;;
    hgnet-b0-sed-v10c)         export_hgnet_b0_sed_v10c ;;
    hgnet-b0-sed-v5j-repro)    export_hgnet_b0_sed_v5j_repro ;;
    hgnet-b0-sed-v5j-pl-v1)   export_hgnet_b0_sed_v5j_pl_v1 ;;
    hgnet-b0-sed-v3-waveform-mix) export_hgnet_b0_sed_v3_waveform_mix ;;
    hgnet-b0-sed-v5j-dualval) export_hgnet_b0_sed_v5j_dualval ;;
    hgnet-b0-sed-v9a-10s-att) export_hgnet_b0_sed_v9a_10s_att ;;
    hgnet-b0-sed-v9b-15s-att) export_hgnet_b0_sed_v9b_15s_att ;;
    hgnet-b0-sed-v9c-20s-att) export_hgnet_b0_sed_v9c_20s_att ;;
    hgnet-b0-sed-v10a-exp027) export_hgnet_b0_sed_v10a_exp027 ;;
    hgnet-b0-sed-v10b-exp027) export_hgnet_b0_sed_v10b_exp027 ;;
    hgnet-b0-sed-v11b-exp027) export_hgnet_b0_sed_v11b_exp027 ;;
    all-019-ablations)
        export_effnet_b0_sed_v5a
        echo ""
        export_effnet_b0_sed_v5b
        echo ""
        export_effnet_b0_sed_v5c
        echo ""
        export_effnet_b0_sed_v5d
        echo ""
        export_hgnet_b0_sed_v5e
        ;;
    all-017-ablations)
        export_effnet_b0_sed_v4b
        echo ""
        export_effnet_b0_sed_v4d
        echo ""
        export_effnet_b0_sed_v4e
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
