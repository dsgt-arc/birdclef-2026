#!/bin/bash
# Submit SLURM job to precompute per-sample losses + per-frame logits for
# v5j (HGNetV2-B0 + LSEHead, LB 0.906) on the TRAINING SET.
#
# Used by D1 diagnosis (2026-04-26): inspect attention behavior on
# highest-loss-during-training and lowest-loss-during-training examples.
# See user/murilo/analysis/notes/2026-04-26-mixup-implementation-audit.md
# for the audit that motivated D1.
#
# Usage:
#   bash run/precompute_v5j_train.sh                  # defaults
#   bash run/precompute_v5j_train.sh --gpu a100       # override GPU
#   bash run/precompute_v5j_train.sh --dry-run        # preview command
#
# Output (under user/murilo/analysis/data/):
#   - losses_v5j_train.parquet           (per-sample loss + GT prob/rank)
#   - losses_v5j_train_logits.npy        (clipwise logits for explorer)
#   - losses_v5j_train_perframe_logits.npy   (per-frame logits, fp16)
#
# Triage is skipped on train-split runs (val-only).

set -euo pipefail

# --- model-specific paths ---
CONFIG="../experiments/019-sed-ablations/config/hgnet_b0_sed_v5j.yaml"
CHECKPOINT="/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5j/best-epochepoch=19.ckpt"
OUTPUT="data/losses_v5j_train.parquet"
DATA_SPLIT="train"

# --- run ---
cd "$(dirname "$0")/.."

exec bash sbatch/run.sh \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --data-split "$DATA_SPLIT" \
  "$@"
