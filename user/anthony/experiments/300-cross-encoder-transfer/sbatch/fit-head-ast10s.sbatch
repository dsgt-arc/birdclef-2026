#!/bin/bash
#SBATCH --job-name=300-ast10s-head
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 60 -q inferno
#SBATCH --output=logs/ast10s-head-%j.out

# Refit the deployable AST ridge head on the 10 s-geometry cache (focal + soundscape),
# the geometry the Kaggle kernel now uses (6 windows/60 s clip). Three steps:
#   1. held-out focal->soundscape λ sweep (honest transfer; compare to the 5 s 0.6448)
#   2. fit the deployed head on focal+soundscape at the winning λ -> artifacts/head_ast.npz
#   3. in-sample replay sanity check
# The 5 s head is backed up to head_ast_5s.npz first (it is gitignored / already on Kaggle).
set -euo pipefail
mkdir -p logs artifacts
export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
echo "REPO_DIR=$REPO_DIR"
source sbatch/_setup-venv.sh

FEAT="${FEATURES_DIR:-$HOME/scratch/birdclef-2026/features/exp300-deploy/ast}"
echo "FEATURES_DIR=$FEAT"

# Ensure the soundscape parquet is linked into the flat deploy dir (the focal array's
# link step is skipped if it ran before the labeled extraction finished).
python -c "
import yaml
from birdclef_300.extract import link_soundscape_parquet
cfg = yaml.safe_load(open('configs/exp300.yaml'))
link_soundscape_parquet(cfg, 'ast', 'frozen-cls-10s')
"
ls -la "$FEAT" "$FEAT/focal" || true

# 1. held-out λ sweep
python -m birdclef_300.deploy.sweep_lambda \
    --features-dir "$FEAT" \
    --out artifacts/head_ast_10s_lambda_sweep.json
BEST=$(python -c "import json;print(f\"{json.load(open('artifacts/head_ast_10s_lambda_sweep.json'))['best_lambda']:.0f}\")")
echo "BEST_LAMBDA=$BEST"

# 2. back up the 5 s head, then fit the 10 s head on focal+soundscape
[ -f artifacts/head_ast.npz ] && cp -f artifacts/head_ast.npz artifacts/head_ast_5s.npz && echo "backed up 5 s head -> head_ast_5s.npz"
birdclef-300 fit-head \
    --features-dir "$FEAT" \
    --sources focal,soundscape \
    --lambda "${LAMBDA:-$BEST}" \
    --encoder ast \
    --out artifacts/head_ast.npz
echo "done"
