#!/bin/bash
# Submit SLURM job to precompute per-sample losses + per-frame logits + per-frame
# attention pre-activation for v6k (HGNetV2-B0 + AttHead competition-standard,
# LB 0.888) on the TRAINING SET.
#
# Used by D1 diagnosis (2026-04-26): comparator to v5j with REAL learned
# attention weights instead of LSE-implicit. The forward hook on
# AttHead.attention_conv produces an extra `*_perframe_att.npy` containing
# the pre-tanh, pre-softmax attention activation. per_frame_analysis.py
# (--head-type=atthead) and train_side_analysis.py consume both files.
#
# Usage:
#   bash run/precompute_v6k_train.sh                  # defaults
#   bash run/precompute_v6k_train.sh --gpu a100       # override GPU
#   bash run/precompute_v6k_train.sh --dry-run        # preview command
#
# Output (under user/murilo/analysis/data/):
#   - losses_v6k_train.parquet               (per-sample loss + GT prob/rank)
#   - losses_v6k_train_logits.npy            (clipwise logits)
#   - losses_v6k_train_perframe_logits.npy   (per-frame logits, fp16)
#   - losses_v6k_train_perframe_att.npy      (per-frame att pre-activation, fp16)
#
# Triage is skipped on train-split runs (val-only).

set -euo pipefail

# --- model-specific paths ---
CONFIG="../experiments/021-sed-training/config/hgnet_b0_atthead_v6k.yaml"
CHECKPOINT="/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/021-sed-training/checkpoints/2026-hgnet-b0-atthead-v6k/best-epochepoch=19.ckpt"
OUTPUT="data/losses_v6k_train.parquet"
DATA_SPLIT="train"

# --- run ---
cd "$(dirname "$0")/.."

exec bash sbatch/run.sh \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --data-split "$DATA_SPLIT" \
  "$@"
