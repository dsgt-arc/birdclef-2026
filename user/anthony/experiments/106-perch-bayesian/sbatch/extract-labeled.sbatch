#!/bin/bash
#SBATCH --job-name=106-sc-labeled
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 30
#SBATCH -q inferno
#SBATCH --output=logs/extract-labeled-%j.out

# Extract Perch embedding + label for all 66 labeled soundscape files (~739 windows).
# Fast — under 5 minutes.

set -eu
mkdir -p logs

source sbatch/_setup-venv.sh

OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('configs/exp106.yaml'))['output']['base_dir'])")

birdclef-106 extract \
    --config configs/exp106.yaml \
    --source soundscape-labeled \
    --output "$OUTPUT_DIR/soundscape_labeled.parquet"
