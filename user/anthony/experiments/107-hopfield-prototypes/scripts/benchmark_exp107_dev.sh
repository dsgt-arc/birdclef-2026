#!/bin/bash
set -euo pipefail

source "$HOME/scratch/birdclef-2026/.venv-107/bin/activate"

DEV_SITES=${DEV_SITES:-S13,S15,S19}
SPLIT_NAME=${SPLIT_NAME:-dev}
PRIMARY_METRIC_NAME=${PRIMARY_METRIC_NAME:-dev_auroc}
LAMBDA_SITE=${LAMBDA_SITE:-0.01}
LAMBDA_XATTN=${LAMBDA_XATTN:-0.01}
LAMBDA_FAM=${LAMBDA_FAM:-0.01}
EPOCHS=${EPOCHS:-20}
PATIENCE=${PATIENCE:-10}
K_SPECIES=${K_SPECIES:-8}
K_SITE=${K_SITE:-8}
D_HIDDEN=${D_HIDDEN:-1024}
D_PROJ=${D_PROJ:-512}
CROSS_ATTENTION_HEADS=${CROSS_ATTENTION_HEADS:-4}
LR=${LR:-5e-4}
BATCH_SIZE=${BATCH_SIZE:-256}
STRONG_STRIDE=${STRONG_STRIDE:-4}
ATTN_DROPOUT=${ATTN_DROPOUT:-0.1}
PROJ_DROPOUT=${PROJ_DROPOUT:-0.1}
DISABLE_SITE_BLEND_DELTA_TRUST=${DISABLE_SITE_BLEND_DELTA_TRUST:-0}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.01}
STRONG_SITE_DROPOUT=${STRONG_SITE_DROPOUT:-0}
STRONG_SITE_DROPOUT_START=${STRONG_SITE_DROPOUT_START:-0}
UNSUP_WARMUP_START=${UNSUP_WARMUP_START:-200}
UNSUP_WARMUP_END=${UNSUP_WARMUP_END:-500}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-2}
SITE_BLEND_LR_MULT=${SITE_BLEND_LR_MULT:-1}
SITE_BLEND_DELTA_LR_MULT=${SITE_BLEND_DELTA_LR_MULT:-}
SITE_BLEND_ANCHOR_LAMBDA=${SITE_BLEND_ANCHOR_LAMBDA:-0}
FOCAL_CACHE_SIZE=${FOCAL_CACHE_SIZE:-32768}
UNSUP_CACHE_SIZE=${UNSUP_CACHE_SIZE:-32768}
UNSUP_BATCH_RECORDINGS=${UNSUP_BATCH_RECORDINGS:-16}
UNSUP_BATCH_WINDOWS_PER_RECORDING=${UNSUP_BATCH_WINDOWS_PER_RECORDING:-8}
VAL_EVERY_STEPS=${VAL_EVERY_STEPS:-100}
SEED=${SEED:-42}
MULTI_SEEDS=${MULTI_SEEDS:-}
CACHE_SUBSET_SEED=${CACHE_SUBSET_SEED:-}
MULTI_CACHE_SUBSET_SEEDS=${MULTI_CACHE_SUBSET_SEEDS:-}
CACHE_ROOT=${CACHE_ROOT:-$HOME/scratch/birdclef-2026/cache/exp107}
INCLUDE_PERCH_LOGITS=${INCLUDE_PERCH_LOGITS:-}
CKPT_SUBDIR=${CKPT_SUBDIR:-exp107-autoresearch-dev-$(date +%Y%m%d-%H%M%S)-$$}
RESULT_ROOT="$HOME/scratch/birdclef-2026/checkpoints/$CKPT_SUBDIR"
RESULT_JSON_SINGLE="$RESULT_ROOT/$SPLIT_NAME/subset_aurocs.json"
RESULT_JSON_LOSO="$RESULT_ROOT/loso_results.json"

mkdir -p "$RESULT_ROOT"
SECONDS=0

perch_flag=()
if [[ -n "$INCLUDE_PERCH_LOGITS" && "$INCLUDE_PERCH_LOGITS" != "0" ]]; then
  perch_flag=(--include-perch-logits)
fi

split_name_flag=()
if [[ "$DEV_SITES" != "all" ]]; then
  split_name_flag=(--split-name "$SPLIT_NAME")
fi

disable_delta_trust_flag=()
if [[ "$DISABLE_SITE_BLEND_DELTA_TRUST" != "0" ]]; then
  disable_delta_trust_flag=(--disable-site-blend-delta-trust)
fi

run_one_seed() {
  local run_seed="$1"
  local run_root="$2"
  local run_cache_subset_seed="${3:-}"
  local cache_subset_seed_flag=()
  if [[ -n "$run_cache_subset_seed" ]]; then
    cache_subset_seed_flag=(--cache-subset-seed "$run_cache_subset_seed")
  fi
  mkdir -p "$run_root"
  mkdir -p "$run_root"
  python -u scripts/train.py \
    --fold "$DEV_SITES" \
    "${split_name_flag[@]}" \
    --lambda-site "$LAMBDA_SITE" \
    --lambda-xattn "$LAMBDA_XATTN" \
    --lambda-fam "$LAMBDA_FAM" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --k-species "$K_SPECIES" \
    --k-site "$K_SITE" \
    --d-hidden "$D_HIDDEN" \
    --d-proj "$D_PROJ" \
    --cross-attention-heads "$CROSS_ATTENTION_HEADS" \
    --lr "$LR" \
    --batch-size "$BATCH_SIZE" \
    --weight-decay "$WEIGHT_DECAY" \
    --site-blend-lr-mult "$SITE_BLEND_LR_MULT" \
    --site-blend-anchor-lambda "$SITE_BLEND_ANCHOR_LAMBDA" \
    --strong-stride "$STRONG_STRIDE" \
    --attn-dropout "$ATTN_DROPOUT" \
    --proj-dropout "$PROJ_DROPOUT" \
    "${disable_delta_trust_flag[@]}" \
    --label-smoothing "$LABEL_SMOOTHING" \
    --strong-site-dropout "$STRONG_SITE_DROPOUT" \
    --strong-site-dropout-start "$STRONG_SITE_DROPOUT_START" \
    --unsup-warmup-start "$UNSUP_WARMUP_START" \
    --unsup-warmup-end "$UNSUP_WARMUP_END" \
    --checkpoint-dir "$run_root" \
    --cache-dir "$CACHE_ROOT" \
    "${cache_subset_seed_flag[@]}" \
    --focal-cache-size "$FOCAL_CACHE_SIZE" \
    --unsup-cache-size "$UNSUP_CACHE_SIZE" \
    --unsup-batch-recordings "$UNSUP_BATCH_RECORDINGS" \
    --unsup-batch-windows-per-recording "$UNSUP_BATCH_WINDOWS_PER_RECORDING" \
    --val-every-steps "$VAL_EVERY_STEPS" \
    --seed "$run_seed" \
    "${perch_flag[@]}"
}

result_json_for_root() {
  local run_root="$1"
  if [[ "$DEV_SITES" == "all" ]]; then
    echo "$run_root/loso_results.json"
  else
    echo "$run_root/$SPLIT_NAME/subset_aurocs.json"
  fi
}

if [[ -n "$MULTI_SEEDS" ]]; then
  IFS=',' read -r -a seed_list <<< "$MULTI_SEEDS"
  if [[ -n "$MULTI_CACHE_SUBSET_SEEDS" ]]; then
    IFS=',' read -r -a subset_seed_list <<< "$MULTI_CACHE_SUBSET_SEEDS"
  else
    subset_seed_list=("$CACHE_SUBSET_SEED")
  fi
  result_paths=()
  for subset_seed in "${subset_seed_list[@]}"; do
    subset_tag="${subset_seed:-default}"
    for run_seed in "${seed_list[@]}"; do
      seed_root="$RESULT_ROOT/subset-$subset_tag/seed-$run_seed"
      run_one_seed "$run_seed" "$seed_root" "$subset_seed"
      result_paths+=("$(result_json_for_root "$seed_root")")
    done
  done
  printf 'result_json=%s\n' "${result_paths[*]}"
  python - "$PRIMARY_METRIC_NAME" "$DEV_SITES" "${result_paths[@]}" <<'PY'
import json
import statistics as stats
import sys
from pathlib import Path

metric_name = sys.argv[1]
dev_sites = sys.argv[2]
paths = [Path(p).expanduser() for p in sys.argv[3:]]
if not paths:
    raise SystemExit("No result paths provided")
rows = [json.loads(p.read_text()) for p in paths]

def mean(xs):
    return float(sum(xs) / len(xs))

if dev_sites == "all":
    primary = [r['pooled_auroc'] for r in rows]
    print(f"METRIC {metric_name}={mean(primary)}")
    for key in ["without_focal", "with_focal", "bird", "amphibian", "insect"]:
        vals = [r.get('subsets', {}).get(key) for r in rows if key in r.get('subsets', {})]
        if vals:
            print(f"METRIC {key}_auroc={mean(vals)}")
    s22_vals = [r.get('per_fold', {}).get('S22', {}).get('auroc') for r in rows if r.get('per_fold', {}).get('S22')]
    if s22_vals:
        print(f"METRIC s22_fold_auroc={mean(s22_vals)}")
    print(f"METRIC n_val_windows={rows[0].get('n_windows', 0)}")
    print(f"METRIC n_folds={rows[0].get('n_folds', 0)}")
else:
    primary = [r['overall_auroc'] for r in rows]
    print(f"METRIC {metric_name}={mean(primary)}")
    for key in ["without_focal", "with_focal", "bird", "amphibian", "insect"]:
        vals = [r.get('subset_auroc', {}).get(key) for r in rows if key in r.get('subset_auroc', {})]
        if vals:
            print(f"METRIC {key}_auroc={mean(vals)}")
    best_steps = [r['best_step'] for r in rows]
    print(f"METRIC best_step={mean(best_steps)}")
    print(f"METRIC n_val_windows={rows[0].get('n_windows', 0)}")
PY
else
  run_one_seed "$SEED" "$RESULT_ROOT" "$CACHE_SUBSET_SEED"
  RESULT_JSON="$(result_json_for_root "$RESULT_ROOT")"
  echo "result_json=$RESULT_JSON"
  python - "$RESULT_JSON" "$PRIMARY_METRIC_NAME" "$DEV_SITES" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser()
metric_name = sys.argv[2]
dev_sites = sys.argv[3]
if not path.exists():
    raise SystemExit(f"Missing result file: {path}")

data = json.loads(path.read_text())
if dev_sites == "all":
    print(f"METRIC {metric_name}={data['pooled_auroc']}")
    for key in ["without_focal", "with_focal", "bird", "amphibian", "insect"]:
        if key in data.get("subsets", {}):
            print(f"METRIC {key}_auroc={data['subsets'][key]}")
    if "S22" in data.get("per_fold", {}):
        print(f"METRIC s22_fold_auroc={data['per_fold']['S22']['auroc']}")
    print(f"METRIC n_val_windows={data.get('n_windows', 0)}")
    print(f"METRIC n_folds={data.get('n_folds', 0)}")
else:
    subset = data.get("subset_auroc", {})
    print(f"METRIC {metric_name}={data['overall_auroc']}")
    for key in ["without_focal", "with_focal", "bird", "amphibian", "insect"]:
        if key in subset:
            print(f"METRIC {key}_auroc={subset[key]}")
    print(f"METRIC best_step={data['best_step']}")
    print(f"METRIC n_val_windows={data.get('n_windows', 0)}")
PY
fi

printf 'METRIC benchmark_seconds=%s\n' "$SECONDS"
printf 'cache_root=%s\n' "$CACHE_ROOT"
