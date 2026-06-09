#!/bin/bash
#SBATCH --job-name=300-ingest-perch
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=2 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 15 -q inferno
#SBATCH --output=logs/ingest-perch-%j.out

# Re-key exp 106's Perch embeddings into the unified cache (no recompute).
# Requires exp 106's soundscape_labeled.parquet on scratch.

set -euo pipefail
mkdir -p logs

source sbatch/_setup-venv.sh

birdclef-300 extract \
    --config configs/exp300.yaml \
    --encoder perch \
    --variant "${VARIANT:-frozen}"
