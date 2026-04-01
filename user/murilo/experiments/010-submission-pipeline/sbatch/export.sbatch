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
    effnet-b0-sed-v1|effnet-b0-sed-v2|effnet-b0-sed-v3a|effnet-b0-sed-v3b|effnet-b0-sed-v3c|effnet-b0-sed-v3d|\
    effnet-b0-sed-v4a|effnet-b0-sed-v4b|effnet-b0-sed-v4b-fulldata|effnet-b0-sed-v4c|effnet-b0-sed-v4d|effnet-b0-sed-v4e|effnet-b0-sed-v4f|effnet-b0-sed-v4g|effnet-b0-sed-v4h|effnet-b0-sed-v4j|\
    effnet-b0-sed-v5a|effnet-b0-sed-v5b|effnet-b0-sed-v5c|effnet-b0-sed-v5d|hgnet-b0-sed-v5e|effnet-b0-sed-v5f|hgnet-b0-sed-v5g|hgnet-b0-sed-v5g-4fold|hgnet-b0-sed-v5h|hgnet-b0-sed-v5i|hgnet-b0-sed-v5j|hgnet-b0-sed-v5j-4fold|hgnet-b0-sed-v5k|hgnet-b0-sed-v5l|effnet-b0-sed-v5m|all-019-ablations|\
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
    hgnet-b0-sed-v5g)          export_hgnet_b0_sed_v5g ;;
    hgnet-b0-sed-v5g-4fold)    export_hgnet_b0_sed_v5g_4fold ;;
    hgnet-b0-sed-v5h)          export_hgnet_b0_sed_v5h ;;
    hgnet-b0-sed-v5k)          export_hgnet_b0_sed_v5k ;;
    hgnet-b0-sed-v5i)          export_hgnet_b0_sed_v5i ;;
    hgnet-b0-sed-v5j)          export_hgnet_b0_sed_v5j ;;
    hgnet-b0-sed-v5j-4fold)    export_hgnet_b0_sed_v5j_4fold ;;
    hgnet-b0-sed-v5l)          export_hgnet_b0_sed_v5l ;;
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
