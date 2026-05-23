#!/bin/bash
# Wrapper for v5j-original LB 0.906 reproduction attempt.
#
# Submits the v5j-era code training (same as v5j-eracode) constrained to
# the same SLURM node as v5j-original (atl1-1-02-004-35-0). Controls for
# GPU fp non-determinism — the only known uncontrolled variable left.
#
# Usage:
#   cd /storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-v5j-era/user/murilo/experiments/019-perch-guided-sed
#   bash sbatch/run_samenode_reprod.sh

set -eu

CONFIG="config/hgnet_b0_sed_v5j_reprod_samenode.yaml"
NODE="atl1-1-02-004-35-0"   # v5j-original's training node
PROJECT_NAME="ps-clef2026_birdclef-0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
RUN_NAME="$(basename "$CONFIG" .yaml)"

if [[ ! -f "${EXPERIMENT_DIR}/${CONFIG}" ]]; then
    echo "Error: Config not found: ${EXPERIMENT_DIR}/${CONFIG}"
    exit 1
fi

LOG_DIR="${EXPERIMENT_DIR}/logs"
mkdir -p "${LOG_DIR}"
JOB_SCRIPT="${SCRIPT_DIR}/_job.sh"

echo "=============================================="
echo "v5j-original LB 0.906 REPRODUCTION ATTEMPT"
echo "=============================================="
echo "Config:    ${CONFIG}"
echo "Run name:  ${RUN_NAME}"
echo "Node:      ${NODE} (same as v5j-original)"
echo "Partition: gpu-rtx6000"
echo ""

sbatch \
    --nodelist="${NODE}" \
    --partition=gpu-rtx6000 \
    --output="${LOG_DIR}/slurm_${RUN_NAME}_%j.log" \
    --export=ALL,CONFIG="${CONFIG}",PROJECT_NAME="${PROJECT_NAME}",FAST_DEV_RUN=false,GPU_CHECK=0,NUM_WORKERS=,FOLD= \
    --job-name="exp019-v5j-reprod-samenode" \
    "${JOB_SCRIPT}"

echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs: ${LOG_DIR}/"
