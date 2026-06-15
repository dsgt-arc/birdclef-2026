#!/bin/bash
#SBATCH --job-name=300-dryrun-abl
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 40 -q inferno
#SBATCH --output=logs/dryrun-abl-%j.out

# Local end-to-end dry-run of submission.py for a NON-Perch encoder: encode a few real
# soundscapes through the kernel's BEATs/BirdNET forward, apply that encoder's head.npz,
# write submission.csv, assert 234 cols / 48 rows / probs in [0,1]. The encoder is read
# from head.npz; we stage the matching head + model dir via env overrides.
#
#   ENCODER=beats   sbatch sbatch/dryrun-kernel-ablation.sbatch
#   ENCODER=birdnet sbatch sbatch/dryrun-kernel-ablation.sbatch

set -euo pipefail
mkdir -p logs
export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
ENCODER=${ENCODER:?set ENCODER=beats|birdnet|ast}

if [ "$ENCODER" = "birdnet" ]; then
    source sbatch/_setup-venv-birdnet.sh
else
    source sbatch/_setup-venv.sh
fi
EXP_DIR="$PWD"
uv pip install librosa pandas >/dev/null 2>&1 || true
# AST: ONNX forward needs onnxruntime; mel still uses transformers' ASTFeatureExtractor.
[ "$ENCODER" = "ast" ] && uv pip install transformers onnxruntime >/dev/null 2>&1 || true

SHARED=/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026
WORK="${TMPDIR:-/tmp}/dryrun-$SLURM_JOB_ID"
mkdir -p "$WORK/test_soundscapes" "$WORK/out" "$WORK/head"

# 4 train soundscapes as stand-in test clips.
i=0
for f in "$SHARED"/train_soundscapes/*.ogg; do
    cp "$f" "$WORK/test_soundscapes/"; i=$((i + 1)); [ "$i" -ge 4 ] && break
done
echo "staged $i soundscapes"

# head.npz = this encoder's head.
ln -sf "$EXP_DIR/artifacts/head_${ENCODER}.npz" "$WORK/head/head.npz"

# Per-encoder model dir.
declare -a EXTRA_ENV=()
if [ "$ENCODER" = "beats" ]; then
    MD="$WORK/beats_md"; mkdir -p "$MD"
    ln -sf "$HOME/arc/birdclef-2026/vendor/unilm/beats" "$MD/beats"
    for pt in "$HOME"/scratch/birdclef-2026/checkpoints/beats/*.pt; do ln -sf "$pt" "$MD/"; done
    EXTRA_ENV=(BEATS_DIR="$MD")
elif [ "$ENCODER" = "ast" ]; then
    # AST runs via ONNX now: point at the exact shipped payload (INT8 graph +
    # preprocessor_config.json), no HF weights. Mirrors the ast-onnx Kaggle dataset.
    EXTRA_ENV=(AST_ONNX_DIR="$EXP_DIR/kaggle_kernel/ast_onnx_dataset")
else
    EXTRA_ENV=(BIRDNET_DIR="$HOME/scratch/birdclef-2026/features/exp300-deploy/birdnet")
fi

cd "$WORK/out"
env "${EXTRA_ENV[@]}" \
    HEAD_DIR="$WORK/head" \
    TEST_SOUNDSCAPES_DIR="$WORK/test_soundscapes" \
    python "$EXP_DIR/kaggle_kernel/submission.py"

echo "=== submission.csv checks ==="
SUB="$WORK/out/predictions/submission.csv" python - <<'PY'
import os, pandas as pd
df = pd.read_csv(os.environ["SUB"])
sp = [c for c in df.columns if c != "row_id"]
v = df[sp].values
print("rows", len(df), "| species cols", len(sp))
print("min", round(float(v.min()),5), "max", round(float(v.max()),5),
      "mean", round(float(v.mean()),5), "| any NaN", bool(df.isna().any().any()))
print("per-row nonzero (mean):", round(float((v > 1e-4).sum(axis=1).mean()), 1))
print("sample row_id", df["row_id"].iloc[0])
assert len(sp) == 234, f"expected 234 cols, got {len(sp)}"
assert len(df) == 48, f"expected 48 rows, got {len(df)}"
assert not df.isna().any().any(), "NaNs"
assert 0.0 <= v.min() and v.max() <= 1.0, "probs out of [0,1]"
assert v.max() > 0.05, "predictions look dead"
print("DRYRUN OK")
PY
