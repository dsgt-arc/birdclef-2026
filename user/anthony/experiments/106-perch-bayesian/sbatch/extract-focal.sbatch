#!/bin/bash
#SBATCH --job-name=106-focal
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=12G
#SBATCH -p cpu-small
#SBATCH -t 360
#SBATCH -q inferno
#SBATCH --array=0-15
#SBATCH --output=logs/extract-focal-%A_%a.out

# Extract Perch embedding + label for focal recordings.
# 35,549 files / 16 shards = ~2,222 files/shard, ~15.5k windows/shard.
# ~30-45 min/shard. Actual RAM ~4 GB.

set -eu
mkdir -p logs

source sbatch/_setup-venv.sh

NUM_SHARDS=16
OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('configs/exp106.yaml'))['output']['base_dir'])")

python -c "from birdclef_106.cli import main; main()" extract \
    --config configs/exp106.yaml \
    --source focal \
    --output "$OUTPUT_DIR/focal/shard-${SLURM_ARRAY_TASK_ID}.parquet" \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --num-shards "$NUM_SHARDS"
