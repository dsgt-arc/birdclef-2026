#!/bin/bash
#SBATCH --job-name=300-focal-birdnet
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 480 -q inferno
#SBATCH --array=0-7
#SBATCH --output=logs/focal-birdnet-%A_%a.out

# Extract BirdNET FOCAL embeddings (deploy-head input) as a CPU array job — each task
# is one shard of train.csv, embedded via bmz/TFLite and pooled 3 s -> 5 s windows.
# Writes {output_dir}/birdnet/focal/shard-NN.parquet + links the soundscape parquet.
#
# Usage (8 shards = the --array size above):
#   sbatch sbatch/extract-focal-birdnet.sbatch
#
# NUM_SHARDS must equal the array width; override both together if you change it.

set -euo pipefail
mkdir -p logs

# Run THIS worktree's code (the focal extractor lives here, maybe not in ~/arc).
export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
echo "REPO_DIR=$REPO_DIR  shard=${SLURM_ARRAY_TASK_ID:-0}/${NUM_SHARDS:-8}"

source sbatch/_setup-venv-birdnet.sh

VARIANT=${VARIANT:-frozen}
NUM_SHARDS=${NUM_SHARDS:-8}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}

birdclef-300 extract \
    --config configs/exp300.yaml \
    --encoder birdnet \
    --variant "$VARIANT" \
    --source focal \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    ${LIMIT:+--limit "$LIMIT"}
