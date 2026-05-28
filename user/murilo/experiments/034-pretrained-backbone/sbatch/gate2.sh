#!/bin/bash
# Exp 034 Phase D — Gate 2 launcher (matches exp 032/033 run.sh pattern).
# Plain bash launcher: submits sbatch/_job.sh with the Gate-2 config.
#
# RUN WITH:
#   cd user/murilo/experiments/034-pretrained-backbone
#   bash sbatch/gate2.sh                                   # configs/gate2-effv2s.yaml, fold 0, 5 ep
#   FOLD=0 bash sbatch/gate2.sh                             # override fold
#   CONFIG=configs/gate2-nfnet.yaml bash sbatch/gate2.sh    # different config
#   bash sbatch/gate2.sh --dry-run                          # preview the sbatch command

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$EXPERIMENT_DIR"                 # → submission dir = experiment dir → $SLURM_SUBMIT_DIR
LOG_DIR="${EXPERIMENT_DIR}/logs"
mkdir -p "$LOG_DIR"

CONFIG="${CONFIG:-configs/gate2-effv2s.yaml}"
FOLD="${FOLD:-0}"
PARTITION="${PARTITION:-gpu-rtx6000,gpu-v100}"
RUN_NAME="$(basename "$CONFIG" .yaml)"

CMD=(sbatch
    --job-name="exp034-${RUN_NAME}"
    --partition="$PARTITION"
    --output="${LOG_DIR}/slurm_${RUN_NAME}-fold${FOLD}_%j.log"
    --export="ALL,CONFIG=${CONFIG},FOLD=${FOLD}"
    "$SCRIPT_DIR/_job.sh")

echo "=============================================="
echo "Experiment 034: Pretrained-backbone — Gate 2"
echo "=============================================="
echo "Config: ${CONFIG}"
echo "Run name: ${RUN_NAME}"
echo "Partition: ${PARTITION}"
echo "Fold: ${FOLD}"
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
