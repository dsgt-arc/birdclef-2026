#!/bin/bash
#SBATCH --job-name=300-dryrun-ast
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 40 -q inferno
#SBATCH --output=logs/dryrun-ast-%j.out

# Local end-to-end dry-run of kaggle_kernel_ast/submission.py PLUS a parity check that
# the kernel's in-process AST forward reproduces the cached cls embedding (cosine ~ 1.0)
# — the key guard for torch-in-kernel deployment. CPU-only.
#
#   sbatch sbatch/dryrun-kernel-ast.sbatch

set -euo pipefail
mkdir -p logs

export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
source sbatch/_setup-venv.sh           # activates .venv-300 (torch/transformers/soundfile) and cd's into the exp dir
EXP_DIR="$PWD"
# Resolve birdclef_300 from THIS worktree (shared-venv editable .pth may point elsewhere).
export PYTHONPATH="$EXP_DIR/src:${PYTHONPATH:-}"

# Build a local HF AST snapshot dir (reuses the HF cache populated during extraction).
AST_SNAPSHOT=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('MIT/ast-finetuned-audioset-10-10-0.4593'))")
echo "AST_SNAPSHOT=$AST_SNAPSHOT"

# ---- Parity check: kernel ASTEncoder vs the cached cls embedding -------------
echo "=== parity check (kernel forward vs cache, cosine ~ 1.0) ==="
AST_SNAPSHOT="$AST_SNAPSHOT" EXP_DIR="$EXP_DIR" python - <<'PY'
import os, sys, numpy as np, polars as pl
sys.path.insert(0, os.path.join(os.environ["EXP_DIR"], "kaggle_kernel_ast"))
import submission as K
from birdclef_300 import audio as A

cache = os.path.expanduser(
    "~/scratch/birdclef-2026/features/exp300/embeddings/"
    "encoder=ast/variant=frozen-cls/source=soundscape-labeled/data.parquet"
)
df = pl.read_parquet(cache, columns=["filename", "start_sec", "embedding"]).head(3)
sdir = "/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/train_soundscapes"

enc = K.ASTEncoder(K.find_ast_snapshot())
cosines = []
for row in df.iter_rows(named=True):
    wav = A.load_audio_16k(os.path.join(sdir, row["filename"]))
    start = int(row["start_sec"] * A.ENCODER_SR)
    win = A.fit_window(wav[start:start + A.ENCODER_WINDOW_SAMPLES])[None, :]
    emb = enc.encode_windows(win)[0].astype(np.float64)
    cached = np.asarray(row["embedding"], dtype=np.float64)
    cos = float(emb @ cached / (np.linalg.norm(emb) * np.linalg.norm(cached) + 1e-12))
    cosines.append(cos)
    print(f"  {row['filename']} @ {row['start_sec']}s: cosine={cos:.6f}")
mn = min(cosines)
print(f"min cosine = {mn:.6f}")
assert mn > 0.999, f"kernel<->cache AST embedding drift: min cosine {mn:.6f} < 0.999"
print("PARITY OK")
PY

# ---- Full submission dry-run (only if the head has been fit) -----------------
HEAD="$EXP_DIR/artifacts/head_ast.npz"
if [ ! -f "$HEAD" ]; then
    echo "NOTE: $HEAD missing — skipping full submission run (fit the head first). Parity OK is sufficient to proceed."
    exit 0
fi

SHARED=/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026
WORK="${TMPDIR:-/tmp}/dryrun-ast-$SLURM_JOB_ID"
mkdir -p "$WORK/test_soundscapes" "$WORK/out"
i=0
for f in "$SHARED"/train_soundscapes/*.ogg; do
    cp "$f" "$WORK/test_soundscapes/"; i=$((i + 1)); [ "$i" -ge 4 ] && break
done
echo "staged $i soundscapes in $WORK/test_soundscapes"

cd "$WORK/out"
AST_SNAPSHOT="$AST_SNAPSHOT" \
HEAD_DIR="$EXP_DIR/artifacts" \
TEST_SOUNDSCAPES_DIR="$WORK/test_soundscapes" \
python "$EXP_DIR/kaggle_kernel_ast/submission.py"

echo "=== submission.csv checks (raw predictions) ==="
SUB="$WORK/out/predictions/submission.csv" python - <<'PY'
import os, pandas as pd
df = pd.read_csv(os.environ["SUB"])
sp = [c for c in df.columns if c != "row_id"]
v = df[sp].values
print("rows", len(df), "| species cols", len(sp))
print("min", round(float(v.min()), 5), "max", round(float(v.max()), 5),
      "mean", round(float(v.mean()), 5), "| any NaN", bool(df.isna().any().any()))
assert len(sp) == 234, f"expected 234 species cols, got {len(sp)}"
assert len(df) == 48, f"expected 48 rows (4 clips x 12 windows), got {len(df)}"
assert not df.isna().any().any(), "NaNs in submission"
assert 0.0 <= v.min() and v.max() <= 1.0, "probs out of [0,1]"
assert v.max() > 0.05, "predictions look dead (max <= 0.05)"
print("DRYRUN OK")
PY
