#!/bin/bash
#SBATCH --job-name=106-focal
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 360
#SBATCH -q inferno
#SBATCH --array=0-7
#SBATCH --output=logs/extract-focal-%A_%a.out

# Extract Perch embedding + label for focal recordings.
# 35,549 files / 8 shards = ~4,444 files/shard.
# Variable duration per file (mean ~35s = ~7 windows), so ~31k windows/shard.
# ~60-90 min/shard.

set -eu
mkdir -p logs

source sbatch/_setup-venv.sh

NUM_SHARDS=8
OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('configs/exp106.yaml'))['output']['base_dir'])")

birdclef-106 extract \
    --config configs/exp106.yaml \
    --source focal \
    --output "$OUTPUT_DIR/focal/shard-${SLURM_ARRAY_TASK_ID}.parquet" \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --num-shards "$NUM_SHARDS"
