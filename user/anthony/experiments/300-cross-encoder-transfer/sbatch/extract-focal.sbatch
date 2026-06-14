#!/bin/bash
#SBATCH --job-name=300-extract-focal
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=10G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -p gpu-rtx6000
#SBATCH -t 240 -q inferno
#SBATCH --array=0-15
#SBATCH --output=logs/extract-focal-%A_%a.out

# Extract a torch encoder's FOCAL embeddings (16-way shard array) into the deploy-head
# features dir laid out as <OUT_BASE>/focal/shard-<i>.parquet. This is the focal side of
# the focal+soundscape cache that deploy.fit_head needs — exp 300 only extracted the 739
# soundscape-labeled windows per encoder, so this fills the gap to deploy a non-Perch
# encoder like Perch. Shard 0 also assembles <OUT_BASE>/soundscape_labeled.parquet from
# the already-extracted labeled cache, completing the dir fit_head reads.
#
# Usage (AST cls deploy):
#   ENCODER=ast VARIANT=frozen-cls POOLING=cls sbatch sbatch/extract-focal.sbatch

set -euo pipefail
mkdir -p logs

module load cuda/12.6.1

# Run THIS worktree's code (new extract-focal CLI + window_audio) regardless of where
# ~/arc points — mirror fit-head.sbatch's REPO_DIR derivation.
export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
echo "REPO_DIR=$REPO_DIR  shard=$SLURM_ARRAY_TASK_ID"
source sbatch/_setup-venv.sh
# The shared .venv-300 editable install of birdclef_300 may point at a SIBLING worktree
# (exp300 vs 300b race over one venv). Prepend THIS worktree's src so the new
# extract-focal CLI + window_audio resolve here, regardless of the editable .pth.
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

ENCODER=${ENCODER:?set ENCODER=ast|beats|eat|ssamba_hf}
VARIANT=${VARIANT:-frozen}
NUM_SHARDS=${NUM_SHARDS:-16}
OUT_BASE=${OUT_BASE:-$HOME/scratch/birdclef-2026/features/exp300-$ENCODER}

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 2)" \
    || { echo 'FATAL: CUDA not available' >&2; exit 2; }

birdclef-300 extract-focal \
    --config configs/exp300.yaml \
    --encoder "$ENCODER" \
    --variant "$VARIANT" \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --num-shards "$NUM_SHARDS" \
    --out-base "$OUT_BASE" \
    ${LIMIT:+--limit "$LIMIT"} \
    ${POOLING:+--pooling "$POOLING"}

# Shard 0 assembles the soundscape side of the deploy cache: copy the already-extracted
# labeled-soundscape parquet (same encoder+variant) into the dir fit_head reads. It only
# needs filename/ground_truth/embedding, which that parquet already carries.
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
    SRC="$HOME/scratch/birdclef-2026/features/exp300/embeddings/encoder=$ENCODER/variant=$VARIANT/source=soundscape-labeled/data.parquet"
    DST="$OUT_BASE/soundscape_labeled.parquet"
    if [ -f "$SRC" ]; then
        cp "$SRC" "$DST"
        echo "assembled $DST from $SRC"
    else
        echo "WARNING: labeled parquet not found at $SRC — assemble soundscape_labeled.parquet manually" >&2
    fi
fi
