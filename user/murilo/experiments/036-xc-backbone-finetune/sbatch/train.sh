#!/bin/bash
# Exp 036 train launcher (matches exp 032/033/034 run.sh + _job.sh pattern).
# Plain bash launcher: submits sbatch/_job.sh from the experiment dir ($SLURM_SUBMIT_DIR).
#
# RUN WITH (from user/murilo/experiments/036-xc-backbone-finetune):
#   bash sbatch/train.sh                                          # A0 parity (a0-fullft-repro)
#   CONFIG=configs/a1-lower-lr.yaml         bash sbatch/train.sh  # Phase A
#   CONFIG=configs/a2-discriminative-lr.yaml bash sbatch/train.sh
#   CONFIG=configs/a3-gradual-unfreeze.yaml  bash sbatch/train.sh
#   CONFIG=configs/b1-class-balanced.yaml    bash sbatch/train.sh # Phase B (after picking the A winner)
#   FOLD=1 CONFIG=configs/a2-discriminative-lr.yaml bash sbatch/train.sh
#   bash sbatch/train.sh --dry-run
#
# After a run (save_top_k=-1): SWA via
#   python scripts/swa_average.py --ckpt-dir <ckpt>/<run>/fold0 --last-k 5 --output <ckpt>/<run>/fold0/swa-last5.ckpt

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$EXPERIMENT_DIR"
LOG_DIR="${EXPERIMENT_DIR}/logs"
mkdir -p "$LOG_DIR"

CONFIG="${CONFIG:-configs/a0-fullft-repro.yaml}"
FOLD="${FOLD:-0}"
PARTITION="${PARTITION:-gpu-rtx6000,gpu-v100}"
FAST_DEV_RUN="${FAST_DEV_RUN:-false}"
RUN_NAME="$(basename "$CONFIG" .yaml)"

CMD=(sbatch
    --job-name "exp036-${RUN_NAME}-fold${FOLD}"
    --partition "$PARTITION"
    --output "${LOG_DIR}/slurm_${RUN_NAME}-fold${FOLD}_%j.log"
    --export "ALL,CONFIG=${CONFIG},FOLD=${FOLD},FAST_DEV_RUN=${FAST_DEV_RUN}"
    "${SCRIPT_DIR}/_job.sh"
)

echo "Config: ${CONFIG}  Fold: ${FOLD}  Partition: ${PARTITION}"
if [ "${1:-}" = "--dry-run" ]; then
    printf '%q ' "${CMD[@]}"; echo
    exit 0
fi
"${CMD[@]}"
