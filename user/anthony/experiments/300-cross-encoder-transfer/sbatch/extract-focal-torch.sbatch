#!/bin/bash
#SBATCH --job-name=300-focal-torch
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=10G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -p gpu-rtx6000
#SBATCH -t 180 -q inferno
#SBATCH --array=0-7
#SBATCH --output=logs/focal-torch-%A_%a.out

# Extract a torch-backend encoder's FOCAL embeddings (deploy-head input) as an array
# job — each task is one shard of train.csv, tiled into 5 s / 16 kHz windows. Writes
# {output_dir}/{encoder}/focal/shard-NN.parquet + links the soundscape parquet.
#
# Usage (8 shards = the --array size above):
#   ENCODER=beats sbatch sbatch/extract-focal-torch.sbatch
#
# NUM_SHARDS must equal the array width; override both together if you change it:
#   ENCODER=beats NUM_SHARDS=4 sbatch --array=0-3 sbatch/extract-focal-torch.sbatch

set -euo pipefail
mkdir -p logs

# Run THIS worktree's code (the focal extractor lives here, maybe not in ~/arc).
export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
echo "REPO_DIR=$REPO_DIR  shard=${SLURM_ARRAY_TASK_ID:-0}/${NUM_SHARDS:-8}"

# BEATs/EAT/SSAMBA load vendored code from $REPO_DIR/vendor/<sub> (git submodules);
# a fresh worktree leaves them empty. Symlink from ~/arc (already initialized) so the
# encoder forward matches the soundscape extraction exactly (no re-clone, no drift).
if [ ! -e "$REPO_DIR/vendor/unilm/beats/BEATs.py" ] && [ -e "$HOME/arc/birdclef-2026/vendor/unilm/beats/BEATs.py" ]; then
    rm -rf "$REPO_DIR/vendor/unilm"
    ln -s "$HOME/arc/birdclef-2026/vendor/unilm" "$REPO_DIR/vendor/unilm"
    echo "linked vendor/unilm -> ~/arc"
fi

module load cuda/12.6.1
source sbatch/_setup-venv.sh

ENCODER=${ENCODER:?set ENCODER=beats|ast|eat|ssamba_hf}
VARIANT=${VARIANT:-frozen}
NUM_SHARDS=${NUM_SHARDS:-8}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 2)" \
    || { echo 'FATAL: CUDA not available' >&2; exit 2; }

birdclef-300 extract \
    --config configs/exp300.yaml \
    --encoder "$ENCODER" \
    --variant "$VARIANT" \
    --source focal \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    ${LIMIT:+--limit "$LIMIT"}
