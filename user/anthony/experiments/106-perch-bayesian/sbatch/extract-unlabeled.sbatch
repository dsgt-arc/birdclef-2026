#!/bin/bash
#SBATCH --job-name=106-sc-unlbl
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 360
#SBATCH -q inferno
#SBATCH --array=0-7
#SBATCH --output=logs/extract-unlabeled-%A_%a.out

# Extract Perch embedding + label for unlabeled soundscapes.
# 10,592 files / 8 shards = ~1,324 files/shard, ~15,888 windows/shard.
# ~45 min/shard at ~6 files/sec.

set -eu
mkdir -p logs

source sbatch/_setup-venv.sh

NUM_SHARDS=8
OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('configs/exp106.yaml'))['output']['base_dir'])")

birdclef-106 extract \
    --config configs/exp106.yaml \
    --source soundscape-unlabeled \
    --output "$OUTPUT_DIR/soundscape_unlabeled/shard-${SLURM_ARRAY_TASK_ID}.parquet" \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --num-shards "$NUM_SHARDS"
