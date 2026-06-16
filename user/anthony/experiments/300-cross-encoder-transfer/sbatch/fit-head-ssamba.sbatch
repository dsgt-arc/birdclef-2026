#!/bin/bash
#SBATCH --job-name=300-ssamba-head
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 60 -q inferno
#SBATCH --output=logs/ssamba-head-%j.out

# Fit the deployable SSAMBA-HF ridge head (SSL ssamba_small_400, mean_patch) on
# focal + soundscape over the 234-species roster. SSAMBA is the only SSL-only encoder
# in the roster (no AudioSet-finetuned checkpoint, unlike EAT/AST). Three steps:
#   1. held-out focal->soundscape λ sweep (honest transfer readout)
#   2. fit the deployed head on focal+soundscape at the winning λ -> artifacts/head_ssamba.npz
#   3. (sweep json carries the in-sample / held-out numbers for results.md)
set -euo pipefail
mkdir -p logs artifacts
export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
echo "REPO_DIR=$REPO_DIR"
source sbatch/_setup-venv.sh

FEAT="${FEATURES_DIR:-$HOME/scratch/birdclef-2026/features/exp300-deploy/ssamba_hf}"
echo "FEATURES_DIR=$FEAT"

# Link the 10 s-geometry soundscape parquet into the flat deploy dir. SSAMBA deploys at
# 10 s (the model pads the mel to 1024 frames regardless, so 10 s windows halve the Kaggle
# forward count); focal + soundscape are both re-extracted at --window-sec 10 (variant
# `frozen-10s`) so the head matches the kernel geometry.
python -c "
import yaml
from birdclef_300.extract import link_soundscape_parquet
cfg = yaml.safe_load(open('configs/exp300.yaml'))
link_soundscape_parquet(cfg, 'ssamba_hf', 'frozen-10s')
"
ls -la "$FEAT" "$FEAT/focal" || true

# Back up the 5 s head before overwriting with the deployed 10 s head.
[ -f artifacts/head_ssamba.npz ] && cp -f artifacts/head_ssamba.npz artifacts/head_ssamba_5s.npz \
    && echo "backed up 5 s head -> head_ssamba_5s.npz"

# 1. held-out λ sweep (focal-only fit, eval on the 739 labeled soundscapes)
python -m birdclef_300.deploy.sweep_lambda \
    --features-dir "$FEAT" \
    --out artifacts/head_ssamba_lambda_sweep.json
BEST=$(python -c "import json;print(f\"{json.load(open('artifacts/head_ssamba_lambda_sweep.json'))['best_lambda']:.0f}\")")
echo "BEST_LAMBDA=$BEST"

# 2. fit the deployed head on focal+soundscape at the winning λ
birdclef-300 fit-head \
    --features-dir "$FEAT" \
    --sources focal,soundscape \
    --lambda "${LAMBDA:-$BEST}" \
    --encoder ssamba_hf \
    --out artifacts/head_ssamba.npz
echo "done"
