#!/bin/bash
#SBATCH -J exp018-perch-extract
#SBATCH -A paceship-clef2026_birdclef
#SBATCH -p cpu-small
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/slurm_perch_extract_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Experiment 018: Perch Ensemble — Extract Embeddings + Train Probes
#
# Runs all offline preparation steps:
#   1. Build species mapping (Perch -> BirdCLEF 2026)
#   2. Extract Perch embeddings + logits from labeled soundscapes
#   3. Train LogReg probes + metadata priors
#   4. Verify and package artifacts for Kaggle
#
# All paths are read from config/perch_ensemble_v1.yaml.
# Embeddings → shared/embeddings/birdclef-2026/perch-v2/labeled/
# Probes     → shared/artifacts/birdclef-2026/perch-ensemble-v1/
#
# Usage:
#   cd user/murilo/experiments/018-perch-ensemble
#   sbatch sbatch/extract_embeddings.sh

set -eu

# ──────────────────────────────────────────────────────────────
# Resolve experiment directory
# ──────────────────────────────────────────────────────────────
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    EXPERIMENT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
fi

CONFIG="${EXPERIMENT_DIR}/config/perch_ensemble_v1.yaml"

echo "=============================================="
echo "Job: Experiment 018 - Perch Embedding Extraction"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Experiment dir: ${EXPERIMENT_DIR}"
echo "Config: ${CONFIG}"
echo ""

# ──────────────────────────────────────────────────────────────
# Setup per-job virtual environment (NEVER in home dir!)
# ──────────────────────────────────────────────────────────────
echo "Setting up per-job virtual environment..."
export UV_CACHE_DIR="$TMPDIR/uv-cache"
JOB_VENV_DIR="$TMPDIR/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"
echo "  Venv: $JOB_VENV"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

echo "  Installing experiment dependencies..."
uv pip install -e "$EXPERIMENT_DIR"

echo "  Installing TensorFlow..."
uv pip install tensorflow

echo "  Virtual environment ready!"
echo ""

python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
echo ""

# ──────────────────────────────────────────────────────────────
# Prepare output directories (shared storage)
# ──────────────────────────────────────────────────────────────
cd "$EXPERIMENT_DIR"
mkdir -p logs artifacts

# Shared storage directories (created by scripts from config, but ensure parents exist)
SHARED_ROOT="/storage/project/ps-clef2026_birdclef-0/shared"
mkdir -p "$SHARED_ROOT/embeddings/birdclef-2026/perch-v2/labeled"
mkdir -p "$SHARED_ROOT/artifacts/birdclef-2026/perch-ensemble-v1"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ──────────────────────────────────────────────────────────────
# Step 1: Build species mapping
# ──────────────────────────────────────────────────────────────
echo "=============================================="
echo "Step 1: Build species mapping"
echo "=============================================="
python scripts/build_species_mapping.py --config "$CONFIG"
echo ""

# ──────────────────────────────────────────────────────────────
# Step 2: Extract Perch embeddings + logits
# ──────────────────────────────────────────────────────────────
echo "=============================================="
echo "Step 2: Extract Perch embeddings (labeled split)"
echo "=============================================="
python scripts/extract_perch_embeddings.py --config "$CONFIG" --split labeled
echo ""

# ──────────────────────────────────────────────────────────────
# Step 3: Train probes + build metadata priors
# ──────────────────────────────────────────────────────────────
echo "=============================================="
echo "Step 3: Train probes + metadata priors"
echo "=============================================="
python scripts/train_probes.py --config "$CONFIG"
echo ""

# ──────────────────────────────────────────────────────────────
# Step 4: Verify and package artifacts
# ──────────────────────────────────────────────────────────────
echo "=============================================="
echo "Step 4: Verify and package artifacts"
echo "=============================================="
python scripts/build_artifacts.py --config "$CONFIG"
echo ""

echo "=============================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "=============================================="
