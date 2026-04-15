#!/bin/bash
set -euo pipefail

source "$HOME/scratch/birdclef-2026/.venv-107/bin/activate"

DEV_SITES=${DEV_SITES:-S13,S15,S19}
LAMBDA_SITE=${LAMBDA_SITE:-0.01}
LAMBDA_XATTN=${LAMBDA_XATTN:-0.01}
LAMBDA_FAM=${LAMBDA_FAM:-0.01}
EPOCHS=${EPOCHS:-20}
LR=${LR:-5e-4}
BATCH_SIZE=${BATCH_SIZE:-256}
STRONG_STRIDE=${STRONG_STRIDE:-4}
ATTN_DROPOUT=${ATTN_DROPOUT:-0.1}
PROJ_DROPOUT=${PROJ_DROPOUT:-0.1}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.01}
UNSUP_WARMUP_START=${UNSUP_WARMUP_START:-200}
UNSUP_WARMUP_END=${UNSUP_WARMUP_END:-500}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-2}
FOCAL_CACHE_SIZE=${FOCAL_CACHE_SIZE:-32768}
UNSUP_CACHE_SIZE=${UNSUP_CACHE_SIZE:-32768}
UNSUP_BATCH_RECORDINGS=${UNSUP_BATCH_RECORDINGS:-16}
UNSUP_BATCH_WINDOWS_PER_RECORDING=${UNSUP_BATCH_WINDOWS_PER_RECORDING:-8}
SEED=${SEED:-42}
CACHE_ROOT=${CACHE_ROOT:-$HOME/scratch/birdclef-2026/cache/exp107}
INCLUDE_PERCH_LOGITS=${INCLUDE_PERCH_LOGITS:-}
CKPT_SUBDIR=${CKPT_SUBDIR:-exp107-autoresearch-dev-$(date +%Y%m%d-%H%M%S)-$$}
RESULT_ROOT="$HOME/scratch/birdclef-2026/checkpoints/$CKPT_SUBDIR"
RESULT_JSON="$RESULT_ROOT/dev/subset_aurocs.json"

mkdir -p "$RESULT_ROOT"
SECONDS=0

perch_flag=()
if [[ -n "$INCLUDE_PERCH_LOGITS" && "$INCLUDE_PERCH_LOGITS" != "0" ]]; then
  perch_flag=(--include-perch-logits)
fi

python -u scripts/train.py \
  --fold "$DEV_SITES" \
  --split-name dev \
  --lambda-site "$LAMBDA_SITE" \
  --lambda-xattn "$LAMBDA_XATTN" \
  --lambda-fam "$LAMBDA_FAM" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --batch-size "$BATCH_SIZE" \
  --weight-decay "$WEIGHT_DECAY" \
  --strong-stride "$STRONG_STRIDE" \
  --attn-dropout "$ATTN_DROPOUT" \
  --proj-dropout "$PROJ_DROPOUT" \
  --label-smoothing "$LABEL_SMOOTHING" \
  --unsup-warmup-start "$UNSUP_WARMUP_START" \
  --unsup-warmup-end "$UNSUP_WARMUP_END" \
  --checkpoint-dir "$RESULT_ROOT" \
  --cache-dir "$CACHE_ROOT" \
  --focal-cache-size "$FOCAL_CACHE_SIZE" \
  --unsup-cache-size "$UNSUP_CACHE_SIZE" \
  --unsup-batch-recordings "$UNSUP_BATCH_RECORDINGS" \
  --unsup-batch-windows-per-recording "$UNSUP_BATCH_WINDOWS_PER_RECORDING" \
  --seed "$SEED" \
  "${perch_flag[@]}"

echo "result_json=$RESULT_JSON"

python - "$RESULT_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser()
if not path.exists():
    raise SystemExit(f"Missing result file: {path}")

data = json.loads(path.read_text())
subset = data.get("subset_auroc", {})
print(f"METRIC dev_auroc={data['overall_auroc']}")
for key in ["without_focal", "with_focal", "bird", "amphibian", "insect"]:
    if key in subset:
        print(f"METRIC {key}_auroc={subset[key]}")
print(f"METRIC best_step={data['best_step']}")
print(f"METRIC n_val_windows={data.get('n_windows', 0)}")
PY

printf 'METRIC benchmark_seconds=%s\n' "$SECONDS"
printf 'cache_root=%s\n' "$CACHE_ROOT"
