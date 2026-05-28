#!/bin/bash
# Exp 034 Phase E/F/G — train launcher (matches exp 032/033 run.sh pattern).
# Plain bash launcher: submits sbatch/_job.sh with a full-run config.
#   MODE=frozen (default) → configs/xcpretrain-effv2s-frozen.yaml (25 ep, Phase E/F)
#   MODE=fullft           → configs/xcpretrain-effv2s-fullft.yaml (50 ep, Phase G)
#   override entirely with  CONFIG=configs/<other>.yaml
#
# RUN WITH:
#   cd user/murilo/experiments/034-pretrained-backbone
#   bash sbatch/train.sh                                   # Phase E: frozen, fold 0
#   for f in 0 1 2 3 4; do FOLD=$f bash sbatch/train.sh; done   # Phase F: 5-fold
#   MODE=fullft bash sbatch/train.sh                       # Phase G: full fine-tune
#   CONFIG=configs/xcpretrain-nfnet-frozen.yaml bash sbatch/train.sh
#   bash sbatch/train.sh --dry-run

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$EXPERIMENT_DIR"
LOG_DIR="${EXPERIMENT_DIR}/logs"
mkdir -p "$LOG_DIR"

MODE="${MODE:-frozen}"
FOLD="${FOLD:-0}"
PARTITION="${PARTITION:-gpu-rtx6000,gpu-v100}"

if [ -n "${CONFIG:-}" ]; then
    :  # explicit config override via env
elif [ "$MODE" = "frozen" ]; then
    CONFIG="configs/xcpretrain-effv2s-frozen.yaml"
elif [ "$MODE" = "fullft" ]; then
    CONFIG="configs/xcpretrain-effv2s-fullft.yaml"
else
    echo "ERROR: MODE must be 'frozen' or 'fullft' (or set CONFIG=...); got '$MODE'" >&2
    exit 1
fi
RUN_NAME="$(basename "$CONFIG" .yaml)"

CMD=(sbatch
    --job-name="exp034-${RUN_NAME}"
    --partition="$PARTITION"
    --output="${LOG_DIR}/slurm_${RUN_NAME}-fold${FOLD}_%j.log"
    --export="ALL,CONFIG=${CONFIG},FOLD=${FOLD}"
    "$SCRIPT_DIR/_job.sh")

echo "=============================================="
echo "Experiment 034: Pretrained-backbone — train"
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
