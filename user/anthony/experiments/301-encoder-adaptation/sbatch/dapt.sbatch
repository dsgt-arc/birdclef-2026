#!/bin/bash
#SBATCH --job-name=301-dapt
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=10G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -p gpu-rtx6000
#SBATCH -t 720 -q inferno
#SBATCH --output=logs/dapt-%j.out

# Soundscape DAPT (continued SSL) for EAT or SSAMBA-HF.
#
# Usage:
#   CONFIG=configs/dapt-eat.yaml    sbatch sbatch/dapt.sbatch
#   CONFIG=configs/dapt-ssamba.yaml sbatch sbatch/dapt.sbatch

set -euo pipefail
mkdir -p logs

module load cuda/12.6.1
source sbatch/_setup-venv.sh

CONFIG=${CONFIG:?set CONFIG=configs/dapt-<encoder>.yaml}
NAME=$(basename "$CONFIG" .yaml)
OUTDIR=${OUTDIR:-artifacts/checkpoints/$NAME}

birdclef-301 dapt --config "$CONFIG" --output-dir "$OUTDIR"
