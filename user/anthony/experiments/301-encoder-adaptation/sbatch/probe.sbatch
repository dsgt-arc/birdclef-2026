#!/bin/bash
#SBATCH --job-name=301-probe
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 60 -q inferno
#SBATCH --output=logs/probe-%j.out

# Ridge LOSO probe over the ADAPTED variants (delegates to the exp 300 probe).
# Compare its macro AUROC against the frozen baseline (exp 300 probe_metrics_frozen.json).
#
# Usage:
#   VARIANT=lora-r8 ENCODERS=beats,eat,ssamba_hf sbatch sbatch/probe.sbatch
#   VARIANT=dapt-ep1 ENCODERS=eat,ssamba_hf       sbatch sbatch/probe.sbatch

set -euo pipefail
mkdir -p logs

source sbatch/_setup-venv.sh

EXP300_CONFIG=${EXP300_CONFIG:-../300-cross-encoder-transfer/configs/exp300.yaml}

birdclef-300 probe \
    --config "$EXP300_CONFIG" \
    --encoders "${ENCODERS:?set ENCODERS=beats,eat,ssamba_hf}" \
    --variant "${VARIANT:?set VARIANT=lora-r8|dapt-ep1}" \
    --artifacts-dir artifacts \
    --workers "${WORKERS:-4}"
