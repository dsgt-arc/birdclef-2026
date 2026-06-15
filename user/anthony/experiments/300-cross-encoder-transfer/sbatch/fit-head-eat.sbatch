#!/bin/bash
#SBATCH --job-name=300-eat-head
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 60 -q inferno
#SBATCH --output=logs/eat-head-%j.out

# Fit the deployable EAT ridge head (AudioSet-finetuned checkpoint, frozen CLS) on
# focal + soundscape over the 234-species roster. EAT is the apples-to-apples frozen
# transformer path (supervised-finetuned like AST/BEATs). Three steps:
#   1. held-out focal->soundscape λ sweep (honest transfer readout)
#   2. fit the deployed head on focal+soundscape at the winning λ -> artifacts/head_eat.npz
#   3. (sweep json carries the in-sample / held-out numbers for results.md)
set -euo pipefail
mkdir -p logs artifacts
export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
echo "REPO_DIR=$REPO_DIR"
source sbatch/_setup-venv.sh

FEAT="${FEATURES_DIR:-$HOME/scratch/birdclef-2026/features/exp300-deploy/eat}"
echo "FEATURES_DIR=$FEAT"

# Ensure the soundscape parquet is linked into the flat deploy dir (idempotent guard;
# the focal array's link step is skipped if it ran before labeled extraction finished).
python -c "
import yaml
from birdclef_300.extract import link_soundscape_parquet
cfg = yaml.safe_load(open('configs/exp300.yaml'))
link_soundscape_parquet(cfg, 'eat', 'frozen-cls-ft')
"
ls -la "$FEAT" "$FEAT/focal" || true

# 1. held-out λ sweep (focal-only fit, eval on the 739 labeled soundscapes)
python -m birdclef_300.deploy.sweep_lambda \
    --features-dir "$FEAT" \
    --out artifacts/head_eat_lambda_sweep.json
BEST=$(python -c "import json;print(f\"{json.load(open('artifacts/head_eat_lambda_sweep.json'))['best_lambda']:.0f}\")")
echo "BEST_LAMBDA=$BEST"

# 2. fit the deployed head on focal+soundscape at the winning λ
birdclef-300 fit-head \
    --features-dir "$FEAT" \
    --sources focal,soundscape \
    --lambda "${LAMBDA:-$BEST}" \
    --encoder eat \
    --out artifacts/head_eat.npz
echo "done"
