#!/bin/bash
# Run the D1 cross-model analysis (v5j vs v6k on training set).
#
# Prerequisite: both run/precompute_v5j_train.sh and run/precompute_v6k_train.sh
# SLURM jobs must have completed and produced their parquet + .npy artifacts
# under user/murilo/analysis/data/.
#
# Local CPU only, ~5 min runtime.
#
# Usage:
#   bash run/d1_train_side_analysis.sh                  # defaults
#   bash run/d1_train_side_analysis.sh --top-pct 5      # override
#
# Output (under user/murilo/analysis/data/d1_train_side/):
#   - v5j_train_enriched.parquet                  (per-sample metrics + bucket col)
#   - v6k_train_enriched.parquet                  (same for v6k)
#   - v5j_summary_by_bucket.csv                   (LSEHead per-bucket stats)
#   - v6k_summary_by_bucket.csv                   (AttHead per-bucket stats)
#   - loss_bucket_crosstab.csv                    (failure-overlap cross-tab)
#   - same_sample_attention_comparison.csv        (head-level discriminator)

set -euo pipefail

# --- artifact paths (produced by the precompute SLURM jobs) ---
V5J_LOSSES="data/losses_v5j_train.parquet"
V5J_PERFRAME="data/losses_v5j_train_perframe_logits.npy"
V6K_LOSSES="data/losses_v6k_train.parquet"
V6K_PERFRAME="data/losses_v6k_train_perframe_logits.npy"
V6K_PERFRAME_ATT="data/losses_v6k_train_perframe_att.npy"

# --- defaults ---
OUTPUT_DIR="data/d1_train_side/"

# --- find project venv ---
PYTHON="${PYTHON:-/storage/home/hcoda1/9/mgustineli3/clef/birdclef-2026/.venv/bin/python}"

# --- run ---
cd "$(dirname "$0")/.."

exec "$PYTHON" train_side_analysis.py \
  --v5j-losses "$V5J_LOSSES" \
  --v5j-perframe "$V5J_PERFRAME" \
  --v6k-losses "$V6K_LOSSES" \
  --v6k-perframe "$V6K_PERFRAME" \
  --v6k-perframe-att "$V6K_PERFRAME_ATT" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
