#!/bin/bash
#SBATCH --job-name=exp108-dry-ridge
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail

exp108_dir="$HOME/arc/birdclef-2026/user/anthony/experiments/108-kaggle-submissions"
cd "$exp108_dir"
mkdir -p logs

source sbatch/_setup-venv.sh

shared="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026"
ridge_weights="$HOME/arc/birdclef-2026/user/anthony/experiments/106-perch-bayesian/artifacts/ridge_model.npz"

# Use the first 3 training soundscapes as a stand-in for Kaggle test_soundscapes/*.ogg.
# Real Kaggle test data is only available inside the scoring kernel.
dry_dir="$(mktemp -d)"
for f in $(ls "$shared/train_soundscapes/" | head -3); do
    ln -s "$shared/train_soundscapes/$f" "$dry_dir/$f"
done
echo "=== dry-run audio dir: $dry_dir ==="
ls "$dry_dir"

out="$exp108_dir/submissions/ridge/dry_run_submission.csv"

python submissions/ridge/infer.py \
    --audio-dir "$dry_dir" \
    --ridge-weights "$ridge_weights" \
    --output "$out" \
    --batch-size 12

echo "=== Dry-run sanity checks ==="
python -c "
import pandas as pd, numpy as np
df = pd.read_csv('$out')
print('shape:', df.shape)
print('columns[:3]:', list(df.columns[:3]))
print('row_id head:', df['row_id'].head().tolist())
print('n unique row_ids:', df['row_id'].nunique())
assert df.shape == (3 * 12, 1 + 234), f'unexpected shape {df.shape}'
scores = df.drop(columns=['row_id']).values
assert not np.isnan(scores).any(), 'NaN in predictions'
assert np.isfinite(scores).all(), 'non-finite in predictions'
print('min:', scores.min(), 'max:', scores.max(), 'mean:', scores.mean())
print('OK')
"

rm -rf "$dry_dir"
