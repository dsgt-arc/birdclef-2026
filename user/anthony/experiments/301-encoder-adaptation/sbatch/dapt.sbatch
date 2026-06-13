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
# CMD selects the objective:
#   dapt         — generic latent-bootstrap (baseline, dapt-<enc>.yaml)
#   dapt-native  — native objective (SSAMBA MPG/MPC | EAT data2vec, dapt-<enc>-native*.yaml)
#
# Usage:
#   CMD=dapt-native CONFIG=configs/dapt-eat-native.yaml       sbatch sbatch/dapt.sbatch
#   CMD=dapt-native CONFIG=configs/dapt-ssamba-native.yaml    sbatch sbatch/dapt.sbatch
#   CMD=dapt-native CONFIG=configs/dapt-eat-native-lora.yaml  sbatch sbatch/dapt.sbatch
#   MAX_STEPS=2 ... sbatch sbatch/dapt.sbatch    # smoke a native path cheaply

set -euo pipefail
mkdir -p logs

module load cuda/12.6.1
source sbatch/_setup-venv.sh

CONFIG=${CONFIG:?set CONFIG=configs/dapt-<encoder>.yaml}
CMD=${CMD:-dapt}
NAME=$(basename "$CONFIG" .yaml)
OUTDIR=${OUTDIR:-artifacts/checkpoints/$NAME}

birdclef-301 "$CMD" --config "$CONFIG" --output-dir "$OUTDIR" ${MAX_STEPS:+--max-steps "$MAX_STEPS"}
