#!/bin/bash
#SBATCH --job-name=300-birdnet-wheels
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 45 -q inferno
#SBATCH --output=logs/birdnet-wheels-%j.out

# Build the offline wheel set for the BirdNET kernel, targeting Kaggle's base image:
# PYTHON 3.12 (cp312 binaries — cp311 wheels fail with "from versions: none"). Pin the
# leaf packages to the versions our training venv used, let pip resolve their deps, then
# prune everything Kaggle's base already ships (torch/numpy/scipy/pandas/sklearn/... and
# the GPU heavies) so pip pulls those from base and only the niche packages from us.

set -euo pipefail
mkdir -p logs

STAGE="${STAGE:-$HOME/scratch/birdclef-2026/kaggle-stage/birdnet}"
WHEELS="$STAGE/wheels-py312"
rm -rf "$WHEELS"; mkdir -p "$WHEELS"

# Throwaway py3.12 venv just to run pip download (matches Kaggle's interpreter).
VENV="${TMPDIR:-/tmp}/wh312-$SLURM_JOB_ID"
uv venv "$VENV" --python 3.12 --seed
source "$VENV/bin/activate"
python --version

# pip wheel (not download): builds sdist-only deps (docopt, grad-cam) into wheels too,
# so nothing reaches Kaggle as a .tar.gz (pip --no-index rejects sdists: "from versions:
# none"). Binary deps with PyPI wheels are fetched, not rebuilt.
python -m pip wheel --wheel-dir "$WHEELS" \
    bioacoustics-model-zoo==0.12.3 opensoundscape==0.12.1 ai-edge-litert==2.1.5 2>&1 | tail -8

echo "=== pruning packages Kaggle base already provides ==="
( cd "$WHEELS" && rm -f \
    torch-*.whl torchvision-*.whl torchaudio-*.whl triton-*.whl \
    nvidia_*.whl cuda_*.whl cuda-*.whl \
    numpy-*.whl scipy-*.whl pandas-*.whl scikit_learn-*.whl scikit_image-*.whl \
    matplotlib-*.whl librosa-*.whl pillow-*.whl Pillow-*.whl opencv_python-*.whl \
    networkx-*.whl sympy-*.whl mpmath-*.whl numba-*.whl llvmlite-*.whl \
    pyarrow-*.whl tifffile-*.whl imageio-*.whl 2>/dev/null || true )

echo "=== final wheel set (py3.12) ==="
ls -1 "$WHEELS" | wc -l
du -sh "$WHEELS"
echo "--- wheels ---"; ls -1 "$WHEELS"
echo "WHEELS=$WHEELS"
