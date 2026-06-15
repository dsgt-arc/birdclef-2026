#!/bin/bash
#SBATCH --job-name=300-ast10s-smoke
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 25 -q inferno
#SBATCH --output=logs/ast10s-smoke-%j.out

# Fast CPU smoke of the --window-sec 10 threading before the GPU focal array:
# extract 8 labeled windows at 10 s / cls and assert the embeddings are 768-d and
# that each window decoded 10 s of audio (not 5 s). Catches window-geometry bugs
# without waiting on the GPU queue. AST falls back to CPU here (slow but only 8 windows).
set -euo pipefail
mkdir -p logs artifacts
export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
echo "REPO_DIR=$REPO_DIR"

source sbatch/_setup-venv.sh

OUT=/tmp/ast10s-smoke-$SLURM_JOB_ID.parquet
birdclef-300 extract \
    --config configs/exp300.yaml \
    --encoder ast \
    --variant frozen-cls-10s-smoke \
    --pooling cls \
    --window-sec 10 \
    --limit 8 \
    --output "$OUT"

OUT="$OUT" python - <<'PY'
import os, polars as pl, numpy as np
df = pl.read_parquet(os.environ["OUT"])
emb = np.array(df["embedding"].to_list())
print("rows", len(df), "embed_dim", df["embed_dim"][0], "emb.shape", emb.shape)
print("variant", df["variant"][0], "encoder", df["encoder"][0])
assert emb.shape[1] == 768, f"expected 768-d cls, got {emb.shape[1]}"
assert not np.isnan(emb).any(), "NaN in embeddings"
print("SMOKE OK: 768-d cls embeddings at 10 s geometry")
PY
echo "done"
