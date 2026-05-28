#!/bin/bash
# Exp 034 Phase E/F/G — train launcher (matches exp 032/033 run.sh pattern).
# A plain bash launcher: it submits sbatch/_job.sh; it is NOT itself a SLURM script.
#
#   MODE=frozen (default) → freeze backbone, head-only, 25 ep (Phase E first LB, Phase F folds).
#   MODE=fullft           → unfreeze backbone, 50 ep, grad-clip 1.0 (Phase G; Sydorskyi).
#
# RUN WITH:
#   cd user/murilo/experiments/034-pretrained-backbone
#   bash sbatch/train.sh                                  # Phase E: frozen, fold 0, 25 ep
#   MAX_EPOCHS=50 RUN_SUFFIX=-50ep bash sbatch/train.sh    # 25-vs-50 LB ablation
#   for f in 0 1 2 3 4; do FOLD=$f bash sbatch/train.sh; done   # Phase F: 5-fold
#   MODE=fullft bash sbatch/train.sh                      # Phase G: full fine-tune
#   BACKBONE=eca_nfnet_l0 bash sbatch/train.sh             # NFNet-L0
#   WANDB=1 bash sbatch/train.sh                           # also log to W&B
#   bash sbatch/train.sh --dry-run                         # preview

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$EXPERIMENT_DIR"
mkdir -p logs

MODE="${MODE:-frozen}"
BACKBONE="${BACKBONE:-tf_efficientnetv2_s_in21k}"
FOLD="${FOLD:-0}"
PARTITION="${PARTITION:-gpu-rtx6000,gpu-v100}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

if [[ "$MODE" == "frozen" ]]; then
    MAX_EPOCHS="${MAX_EPOCHS:-25}"; FREEZE=1; GRAD_CLIP="${GRAD_CLIP:-0.0}"
elif [[ "$MODE" == "fullft" ]]; then
    MAX_EPOCHS="${MAX_EPOCHS:-50}"; FREEZE=0; GRAD_CLIP="${GRAD_CLIP:-1.0}"
else
    echo "ERROR: MODE must be 'frozen' or 'fullft', got '$MODE'" >&2; exit 1
fi

RUN_NAME="xcpretrain-${BACKBONE}-${MODE}${RUN_SUFFIX}"
LOG_DIR="${EXPERIMENT_DIR}/logs"
EXPORTS="ALL,BACKBONE=$BACKBONE,FOLD=$FOLD,MAX_EPOCHS=$MAX_EPOCHS,GRAD_CLIP=$GRAD_CLIP,RUN_NAME=$RUN_NAME,FREEZE=$FREEZE,LOAD_PRETRAINED=1,SAVE_CKPT=1"
[[ "${WANDB:-0}" == "1" ]] && EXPORTS="$EXPORTS,WANDB=1"

CMD=(sbatch
    --job-name="exp034-${MODE}"
    --partition="$PARTITION"
    --output="logs/slurm-${MODE}-%j.out"
    --export="$EXPORTS"
    "$SCRIPT_DIR/_job.sh")

echo "=============================================="
echo "Experiment 034: Pretrained-backbone — train (${MODE})"
echo "=============================================="
echo "Backbone: ${BACKBONE}"
echo "Run name: ${RUN_NAME}"
echo "Partition: ${PARTITION}"
echo "Fold: ${FOLD}"
echo "Epochs: ${MAX_EPOCHS}   Mode: ${MODE}   grad_clip: ${GRAD_CLIP}"
echo ""

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "[DRY-RUN] ${CMD[*]}"
else
    echo "Submitting job..."
    "${CMD[@]}"
    echo ""
    echo "Monitor: squeue -u \$USER"
    echo "Logs: ${LOG_DIR}/"
fi
