#!/bin/bash
#SBATCH --job-name=300-extract-birdnet
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 60 -q inferno
#SBATCH --output=logs/extract-birdnet-%j.out

# BirdNET anchor embeddings via bioacoustics-model-zoo (separate TF venv).

set -euo pipefail
mkdir -p logs

source sbatch/_setup-venv-birdnet.sh

birdclef-300 extract \
    --config configs/exp300.yaml \
    --encoder birdnet \
    --variant "${VARIANT:-frozen}"
