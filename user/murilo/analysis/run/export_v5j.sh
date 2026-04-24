#!/bin/bash
# Bulk attribution figure export for hgnet-b0-sed-v5j (0.906 LB baseline).
#
# Usage:
#   bash run/export_v5j.sh                      # defaults: top-k 20, source all, cpu
#   bash run/export_v5j.sh --source focal       # override any flag
#   bash run/export_v5j.sh --top-k 50 --device cuda
#
# To add a new model: copy this file, rename, and update the four paths below.

set -euo pipefail

# --- model-specific paths ---
LOSSES="data/losses_v5j.parquet"
LOGITS="data/losses_v5j_logits.npy"
CONFIG="../experiments/019-sed-ablations/config/hgnet_b0_sed_v5j.yaml"
CHECKPOINT="/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5j/best-epochepoch=19.ckpt"

# --- defaults (override via CLI) ---
TOP_K=20
BATCH_K=5
SOURCE="all"
OUTPUT_DIR="figures/"
DEVICE="cpu"

# --- find project venv ---
PYTHON="${PYTHON:-/storage/home/hcoda1/9/mgustineli3/clef/birdclef-2026/.venv/bin/python}"

# --- run ---
cd "$(dirname "$0")/.."

exec "$PYTHON" export_figures.py \
  --losses "$LOSSES" \
  --logits "$LOGITS" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --top-k "$TOP_K" \
  --batch-k "$BATCH_K" \
  --source "$SOURCE" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  "$@"
