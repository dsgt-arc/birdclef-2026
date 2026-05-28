#!/bin/bash
# Exp 034 Phase D — Gate 2 launcher (matches exp 032/033 run.sh pattern).
# A plain bash launcher: it submits sbatch/_job.sh; it is NOT itself a SLURM script.
#
# RUN WITH:
#   cd user/murilo/experiments/034-pretrained-backbone
#   bash sbatch/gate2.sh                          # EffNetV2-S, fold 0, 5 ep, frozen + pretrained
#   BACKBONE=eca_nfnet_l0 bash sbatch/gate2.sh     # NFNet-L0 variant
#   FOLD=0 MAX_EPOCHS=5 bash sbatch/gate2.sh        # overrides
#   bash sbatch/gate2.sh --dry-run                  # preview the sbatch command

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$EXPERIMENT_DIR"                 # → submission dir = experiment dir → $SLURM_SUBMIT_DIR
mkdir -p logs

BACKBONE="${BACKBONE:-tf_efficientnetv2_s_in21k}"
FOLD="${FOLD:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-5}"
PARTITION="${PARTITION:-gpu-rtx6000,gpu-v100}"
RUN_NAME="gate2-${BACKBONE}"
LOG_DIR="${EXPERIMENT_DIR}/logs"

CMD=(sbatch
    --job-name="exp034-gate2"
    --partition="$PARTITION"
    --output="${LOG_DIR}/slurm-gate2-%j.out"
    --export="ALL,BACKBONE=$BACKBONE,FOLD=$FOLD,MAX_EPOCHS=$MAX_EPOCHS,RUN_NAME=$RUN_NAME,FREEZE=1,LOAD_PRETRAINED=1"
    "$SCRIPT_DIR/_job.sh")

echo "=============================================="
echo "Experiment 034: Pretrained-backbone — Gate 2"
echo "=============================================="
echo "Backbone: ${BACKBONE}"
echo "Run name: ${RUN_NAME}"
echo "Partition: ${PARTITION}"
echo "Fold: ${FOLD}"
echo "Epochs: ${MAX_EPOCHS} (frozen-backbone head-only sanity)"
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
