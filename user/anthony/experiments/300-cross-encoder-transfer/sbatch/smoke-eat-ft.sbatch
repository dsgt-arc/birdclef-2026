#!/bin/bash
#SBATCH --job-name=300-eat-ft-smoke
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 30 -q inferno
#SBATCH --output=logs/eat-ft-smoke-%j.out

# Verify the AudioSet-FINETUNED EAT checkpoint loads through our wrapper and that
# extract_features returns 768-d cls embeddings at the 5 s/512-frame geometry, BEFORE
# committing the GPU focal array. The finetuned checkpoint (a classification model) is
# untested with EATEncoder, which calls model.extract_features(mel) — the main risk is a
# different remote-code API or a required input length. Also times the per-window CPU
# forward (deploy cost signal vs the AST CPU wall).
set -euo pipefail
mkdir -p logs
export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
echo "REPO_DIR=$REPO_DIR"
source sbatch/_setup-venv.sh

OUT=/tmp/eat-ft-smoke-$SLURM_JOB_ID.parquet
time birdclef-300 extract \
    --config configs/exp300.yaml \
    --encoder eat \
    --variant frozen-cls-ft-smoke \
    --checkpoint "${CHECKPOINT:-worstchan/EAT-base_epoch30_finetune_AS2M}" \
    --pooling cls \
    --limit 8 \
    --output "$OUT"

OUT="$OUT" python - <<'PY'
import os, polars as pl, numpy as np
df = pl.read_parquet(os.environ["OUT"])
emb = np.array(df["embedding"].to_list())
print("rows", len(df), "embed_dim", df["embed_dim"][0], "emb.shape", emb.shape)
assert emb.shape[1] == 768, f"expected 768-d, got {emb.shape[1]}"
assert not np.isnan(emb).any(), "NaN in embeddings"
assert np.abs(emb).sum() > 0, "all-zero embeddings"
print("SMOKE OK: EAT finetune_AS2M cls 768-d at 5s/512")
PY
echo "done"
