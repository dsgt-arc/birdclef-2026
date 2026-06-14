#!/bin/bash
#SBATCH --job-name=300-stage-birdnet
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 60 -q inferno
#SBATCH --output=logs/stage-birdnet-%j.out

# Stage the two Kaggle datasets the BirdNET kernel needs to run internet-OFF:
#   weights/  <- bmz BirdNET TFLite model (downloaded into BMZ_CACHE here)
#   wheels/   <- bioacoustics-model-zoo + ai-edge-litert wheel closure (offline pip)
# Compute node has internet; the Kaggle kernel does not. Output -> $STAGE on scratch.

set -euo pipefail
mkdir -p logs
export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"

STAGE="${STAGE:-$HOME/scratch/birdclef-2026/kaggle-stage/birdnet}"
WEIGHTS="$STAGE/weights"
WHEELS="$STAGE/wheels"
mkdir -p "$WEIGHTS" "$WHEELS"

# bmz reads BMZ_CACHE for its model cache; point it at our staging dir so the download
# lands exactly where the kernel will look (kernel sets BMZ_CACHE=model_dir).
export BMZ_CACHE="$WEIGHTS"
source sbatch/_setup-venv-birdnet.sh

echo "=== load BirdNET -> download weights into $WEIGHTS ==="
python - <<'PY'
import os, bioacoustics_model_zoo as bmz
m = bmz.list_models()["BirdNET"]()
print("BirdNET loaded; BMZ_CACHE=", os.environ.get("BMZ_CACHE"))
for attr in ("model_path", "tflite_path", "checkpoint_url", "model_url"):
    if hasattr(m, attr):
        print(f"  {attr} = {getattr(m, attr)}")
PY

echo "=== weights staged ==="
find "$WEIGHTS" -maxdepth 3 -type f \( -iname "*.tflite" -o -iname "*.txt" -o -iname "*.csv" -o -iname "*labels*" \) -printf "%s\t%p\n" | sort
du -sh "$WEIGHTS"

echo "=== download bmz + litert wheel closure into $WHEELS ==="
python -m pip download --dest "$WHEELS" \
    --python-version 3.11 --only-binary=:all: \
    bioacoustics-model-zoo ai-edge-litert 2>&1 | tail -15 || \
python -m pip download --dest "$WHEELS" bioacoustics-model-zoo ai-edge-litert 2>&1 | tail -15
echo "=== wheels staged ==="
ls -1 "$WHEELS" | wc -l
du -sh "$WHEELS"
echo "STAGE=$STAGE"
