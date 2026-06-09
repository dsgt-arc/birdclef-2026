#!/bin/bash
#SBATCH --job-name=301-ksshift
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 30 -q inferno
#SBATCH --output=logs/ksshift-%j.out

# KS-shift between an encoder's frozen and adapted embeddings on the labeled
# windows (does adaptation move the representation?).
#
# Usage:
#   ENCODER=eat VARIANT=lora-r8 sbatch sbatch/ksshift.sbatch

set -euo pipefail
mkdir -p logs

source sbatch/_setup-venv.sh

ENCODER=${ENCODER:?set ENCODER=beats|eat|ssamba_hf}
VARIANT=${VARIANT:?set VARIANT=lora-r8|dapt-ep1}
CACHE=${CACHE:-~/scratch/birdclef-2026/features/exp300/embeddings}
CACHE=$(eval echo "$CACHE")
SRC=source=soundscape-labeled/data.parquet

birdclef-301 ksshift \
    --frozen  "$CACHE/encoder=$ENCODER/variant=frozen/$SRC" \
    --adapted "$CACHE/encoder=$ENCODER/variant=$VARIANT/$SRC" \
    --output  "artifacts/ks_${ENCODER}_${VARIANT}.json" \
    --label   "$ENCODER frozen-vs-$VARIANT"
