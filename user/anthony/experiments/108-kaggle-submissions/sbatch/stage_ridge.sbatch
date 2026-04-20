#!/bin/bash
#SBATCH --job-name=exp108-stage-ridge
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail

exp108_dir="$HOME/arc/birdclef-2026/user/anthony/experiments/108-kaggle-submissions"
cd "$exp108_dir"
mkdir -p logs

source sbatch/_setup-venv.sh

echo "=== Staging Kaggle deps wheels ==="
python scripts/stage_deps.py --output-dir artifacts/dist/deps

echo "=== Staging ridge weights dataset ==="
python scripts/stage_ridge_weights.py --output-dir artifacts/dist/weights-ridge

echo "=== Staging complete ==="
ls -la artifacts/dist/
du -sh artifacts/dist/*
