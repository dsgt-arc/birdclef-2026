#!/bin/bash
#SBATCH --job-name=300-dryrun
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 30 -q inferno
#SBATCH --output=logs/dryrun-%j.out

# Local end-to-end dry-run of kaggle_kernel/submission.py: encode a few real
# soundscapes through the SAME Perch ONNX used for training, apply head.npz, and
# write a submission.csv. Validates the inference path before any Kaggle push.

set -euo pipefail
mkdir -p logs

export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
source sbatch/_setup-venv.sh           # activates venv and cd's into the exp dir
EXP_DIR="$PWD"

# Kernel-only deps (not in the shared training venv) + pandas for the assertions.
uv pip install onnxruntime librosa huggingface_hub pandas

SHARED=/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026
WORK="${TMPDIR:-/tmp}/dryrun-$SLURM_JOB_ID"
mkdir -p "$WORK/test_soundscapes" "$WORK/out"

# Grab a few train soundscapes as stand-in test clips.
i=0
for f in "$SHARED"/train_soundscapes/*.ogg; do
    cp "$f" "$WORK/test_soundscapes/"
    i=$((i + 1)); [ "$i" -ge 4 ] && break
done
echo "staged $i soundscapes in $WORK/test_soundscapes"

# Fetch the Perch ONNX once (same graph as the exp 106 training cache).
PERCH_ONNX=$(python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('justinchuby/Perch-onnx', 'perch_v2_no_dft.onnx'))")
echo "PERCH_ONNX=$PERCH_ONNX"

# Run the kernel from $WORK/out so it writes predictions/submission.csv there.
# No SAMPLE_SUBMISSION: stand-in clips are *train* soundscapes whose row_ids don't
# match the test sample_submission, so a merge would zero everything. Emitting raw
# predictions lets us check the actual values; the sample_sub merge path is exercised
# for real on Kaggle (where test filenames DO match) and is trivial reindex logic.
cd "$WORK/out"
PERCH_ONNX="$PERCH_ONNX" \
HEAD_DIR="$EXP_DIR/artifacts" \
TEST_SOUNDSCAPES_DIR="$WORK/test_soundscapes" \
python "$EXP_DIR/kaggle_kernel/submission.py"

echo "=== submission.csv checks (raw predictions) ==="
SUB="$WORK/out/predictions/submission.csv" python - <<'PY'
import os, numpy as np, pandas as pd
df = pd.read_csv(os.environ["SUB"])
sp = [c for c in df.columns if c != "row_id"]
v = df[sp].values
print("rows", len(df), "| species cols", len(sp))
print("min", round(float(v.min()), 5), "max", round(float(v.max()), 5),
      "mean", round(float(v.mean()), 5), "| any NaN", bool(df.isna().any().any()))
print("per-row nonzero species (mean):", round(float((v > 1e-4).sum(axis=1).mean()), 1))
print("sample row_id", df["row_id"].iloc[0])
assert len(sp) == 234, f"expected 234 species cols, got {len(sp)}"
assert len(df) == 48, f"expected 48 rows (4 clips x 12 windows), got {len(df)}"
assert not df.isna().any().any(), "NaNs in submission"
assert 0.0 <= v.min() and v.max() <= 1.0, "probs out of [0,1]"
assert v.max() > 0.05, "predictions look dead (max <= 0.05)"
print("DRYRUN OK")
PY
