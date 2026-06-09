#!/bin/bash
#SBATCH --job-name=301-lora
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=10G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -p gpu-rtx6000
#SBATCH -t 480 -q inferno
#SBATCH --output=logs/lora-%j.out

# Supervised-focal LoRA adaptation for one encoder.
#
# Usage:
#   CONFIG=configs/lora-eat.yaml    sbatch sbatch/lora.sbatch
#   CONFIG=configs/lora-beats.yaml  sbatch sbatch/lora.sbatch
#   CONFIG=configs/lora-ssamba.yaml sbatch sbatch/lora.sbatch

set -euo pipefail
mkdir -p logs

module load cuda/12.6.1
source sbatch/_setup-venv.sh

CONFIG=${CONFIG:?set CONFIG=configs/lora-<encoder>.yaml}
NAME=$(basename "$CONFIG" .yaml)
OUTDIR=${OUTDIR:-artifacts/checkpoints/$NAME}

birdclef-301 lora --config "$CONFIG" --output-dir "$OUTDIR"
