#!/bin/bash
#SBATCH --job-name=300-dryrun-ssamba
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 30 -q inferno
#SBATCH --output=logs/dryrun-ssamba-%j.out

# Local end-to-end dry-run of submission.py for SSAMBA (eager pure-PyTorch, 10 s geometry).
# Validates the glue the consistency check did NOT exercise: head load (encoder=ssamba_hf),
# build_encoder dispatch (SSAMBA_WEIGHTS_DIR), and the repeat(2) -> 12-rows/clip 5 s grid.
# 4 vCPU == Kaggle, so the timing here also previews the wall (6 win/clip x 4 = 24 fwd).
#   REPO_DIR=$PWD/../../../.. sbatch sbatch/dryrun-ssamba.sbatch
set -euo pipefail
mkdir -p logs
export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
source sbatch/_setup-venv.sh
EXP_DIR="$PWD"
uv pip install pandas >/dev/null 2>&1

SHARED=/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026
WORK="${TMPDIR:-/tmp}/dryrun-ssamba-$SLURM_JOB_ID"
mkdir -p "$WORK/test_soundscapes" "$WORK/out" "$WORK/head"

# Isolate the SSAMBA head so the kernel's *.npz glob is unambiguous.
cp "$EXP_DIR/artifacts/head_ssamba.npz" "$WORK/head/"

# A few train soundscapes as stand-in test clips.
i=0
for f in "$SHARED"/train_soundscapes/*.ogg; do
    cp "$f" "$WORK/test_soundscapes/"; i=$((i + 1)); [ "$i" -ge 4 ] && break
done
echo "staged $i soundscapes"

cd "$WORK/out"
HEAD_DIR="$WORK/head" \
SSAMBA_WEIGHTS_DIR="$EXP_DIR/kaggle_kernel/ssamba_weights_dataset" \
TEST_SOUNDSCAPES_DIR="$WORK/test_soundscapes" \
python -u "$EXP_DIR/kaggle_kernel/submission.py"

echo "=== submission.csv checks ==="
SUB="$WORK/out/predictions/submission.csv" python - <<'PY'
import os, pandas as pd
df = pd.read_csv(os.environ["SUB"])
sp = [c for c in df.columns if c != "row_id"]
v = df[sp].values
print("rows", len(df), "| species", len(sp),
      "| min", round(float(v.min()), 5), "max", round(float(v.max()), 5),
      "mean", round(float(v.mean()), 5), "| NaN", bool(df.isna().any().any()))
print("sample row_id", df["row_id"].iloc[0], "| last", df["row_id"].iloc[-1])
assert len(sp) == 234, f"expected 234 species, got {len(sp)}"
assert len(df) == 48, f"expected 48 rows (4 clips x 12), got {len(df)}"
assert not df.isna().any().any(), "NaNs"
assert 0.0 <= v.min() and v.max() <= 1.0, "probs out of [0,1]"
assert v.max() > 0.05, "predictions look dead"
print("DRYRUN OK")
PY
