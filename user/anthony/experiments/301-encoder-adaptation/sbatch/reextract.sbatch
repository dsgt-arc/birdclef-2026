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
# Writes the pooled data.parquet (POOLED=1, default) and/or the per-patch
# patches.parquet (PATCH=1, for the proto frame-max readout) in ONE job.
#
# Usage (adapted, both caches):
#   ENCODER=eat VARIANT=dapt-native-ep1 \
#       STATE_DICT=artifacts/checkpoints/dapt-eat-native/state_dict.pt PATCH=1 \
#       sbatch sbatch/reextract.sbatch
# Usage (LoRA):
#   ENCODER=eat VARIANT=dapt-native-lora-ep1 \
#       ADAPTER=artifacts/checkpoints/dapt-eat-native-lora/adapter PATCH=1 \
#       sbatch sbatch/reextract.sbatch
# Usage (frozen patch only — pooled frozen already exists, don't overwrite):
#   ENCODER=eat VARIANT=frozen POOLED=0 PATCH=1 sbatch sbatch/reextract.sbatch

set -euo pipefail
mkdir -p logs

module load cuda/12.6.1
source sbatch/_setup-venv.sh

ENCODER=${ENCODER:?set ENCODER=beats|eat|ssamba_hf}
VARIANT=${VARIANT:?set VARIANT=lora-r8|dapt-native-ep1|...}
EXP300_CONFIG=${EXP300_CONFIG:-../300-cross-encoder-transfer/configs/exp300.yaml}

OVERLAY=()
[ -n "${ADAPTER:-}" ]    && OVERLAY+=(--adapter "$ADAPTER")
[ -n "${STATE_DICT:-}" ] && OVERLAY+=(--state-dict "$STATE_DICT")

BASE=(birdclef-300 extract --config "$EXP300_CONFIG" --encoder "$ENCODER" --variant "$VARIANT")

if [ "${POOLED:-1}" != "0" ]; then
    echo ">>> pooled extract ($ENCODER/$VARIANT)"
    "${BASE[@]}" "${OVERLAY[@]}"
fi
if [ "${PATCH:-0}" != "0" ]; then
    echo ">>> patch extract ($ENCODER/$VARIANT)"
    "${BASE[@]}" "${OVERLAY[@]}" --patch
fi
