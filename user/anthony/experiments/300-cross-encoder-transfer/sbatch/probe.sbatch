#!/bin/bash
#SBATCH --job-name=300-probe
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 60 -q inferno
#SBATCH --output=logs/probe-%j.out

# Ridge LOSO probe over every (encoder, variant) cache -> artifacts/probe_metrics_*.json
#
# Usage:
#   sbatch sbatch/probe.sbatch                         # frozen, full roster
#   VARIANT=lora-r8 ENCODERS=beats,eat,ssamba_hf sbatch sbatch/probe.sbatch

set -euo pipefail
mkdir -p logs

source sbatch/_setup-venv.sh

birdclef-300 probe \
    --config configs/exp300.yaml \
    --encoders "${ENCODERS:-beats,ast,eat,ssamba_hf,perch,birdnet}" \
    --variant "${VARIANT:-frozen}" \
    --workers "${WORKERS:-4}"
