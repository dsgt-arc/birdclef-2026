#!/bin/bash
#SBATCH --job-name=301-reextract
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=10G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -p gpu-rtx6000
#SBATCH -t 120 -q inferno
#SBATCH --output=logs/reextract-%j.out

# Re-extract an adapted encoder's labeled embeddings into the shared exp 300 cache
# by delegating to the exp 300 extractor with the LoRA adapter / DAPT state_dict.
#
# Usage (LoRA):
#   ENCODER=eat VARIANT=lora-r8 ADAPTER=artifacts/checkpoints/lora-eat/adapter \
#       sbatch sbatch/reextract.sbatch
# Usage (DAPT):
#   ENCODER=eat VARIANT=dapt-ep1 STATE_DICT=artifacts/checkpoints/dapt-eat/state_dict.pt \
#       sbatch sbatch/reextract.sbatch

set -euo pipefail
mkdir -p logs

module load cuda/12.6.1
source sbatch/_setup-venv.sh

ENCODER=${ENCODER:?set ENCODER=beats|eat|ssamba_hf}
VARIANT=${VARIANT:?set VARIANT=lora-r8|dapt-ep1|...}
EXP300_CONFIG=${EXP300_CONFIG:-../300-cross-encoder-transfer/configs/exp300.yaml}

OVERLAY=()
[ -n "${ADAPTER:-}" ]    && OVERLAY+=(--adapter "$ADAPTER")
[ -n "${STATE_DICT:-}" ] && OVERLAY+=(--state-dict "$STATE_DICT")

birdclef-300 extract \
    --config "$EXP300_CONFIG" \
    --encoder "$ENCODER" \
    --variant "$VARIANT" \
    "${OVERLAY[@]}"
