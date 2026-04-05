#!/bin/bash
#SBATCH --job-name=105-focal
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 120
#SBATCH -q inferno
#SBATCH --array=0-7
#SBATCH --output=logs/focal-%A_%a.out

# Stage 1 (focal): extract Perch pooled embedding + spatial_embedding for
# every 5s chunk of every focal recording in train.csv. Sharded via SLURM
# array — each task processes 1/8 of the focal set (~4444 recordings, ~31k
# windows). Writes to $SCRATCH/birdclef-2026/features/perch_v2/focal/shard-NNN/.
#
# Usage:
#   cd user/anthony/experiments/105-perch-frame-sed
#   sbatch sbatch/extract-focal.sbatch
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/focal-<JOBID>_0.out   # shard 0 progress

set -eu
mkdir -p logs

source sbatch/_setup-venv.sh

SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-8}

echo ""
echo "=== Stage 1: focal extraction (shard $SHARD_INDEX / $NUM_SHARDS) ==="
birdclef-105 extract \
    --config configs/exp105.yaml \
    --source focal \
    --shard-index "$SHARD_INDEX" \
    --num-shards "$NUM_SHARDS"

echo ""
echo "=== Done ==="
