#!/bin/bash
#SBATCH -J exp018-perch-extract
#SBATCH -A paceship-clef2026_birdclef
#SBATCH -p cpu-small
#SBATCH -q inferno
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/slurm_perch_extract_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Experiment 018: Perch Ensemble — Extract Embeddings + Train Probes
#
# Uses perch_v2_cpu model (same variant as Kaggle) — runs on CPU, no GPU needed.
#
# Usage:
#   cd user/murilo/experiments/018-perch-ensemble
#   sbatch sbatch/extract_embeddings.sh                    # labeled (66 files)
#   sbatch sbatch/extract_embeddings.sh --split unlabeled  # unlabeled (10,592 files)

set -eu

# ──────────────────────────────────────────────────────────────
# Parse arguments (forward --split to extract script)
# ──────────────────────────────────────────────────────────────
SPLIT="labeled"
while [[ $# -gt 0 ]]; do
    case $1 in
        --split)
            SPLIT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

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
echo "Split: ${SPLIT}"
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

# perch_v2_cpu runs on CPU — no CUDA needed, plain tensorflow is fine
echo "  Installing TensorFlow (CPU)..."
uv pip install tensorflow-cpu

echo "  Virtual environment ready!"
echo ""

python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
echo ""

# ──────────────────────────────────────────────────────────────
# Download perch_v2_cpu if not already present
# ──────────────────────────────────────────────────────────────
PERCH_MODEL_DIR="/storage/project/ps-clef2026_birdclef-0/shared/models/perch-v2-cpu"

if [ ! -f "$PERCH_MODEL_DIR/saved_model.pb" ]; then
    echo "=============================================="
    echo "Downloading perch_v2_cpu model..."
    echo "=============================================="
    mkdir -p "$PERCH_MODEL_DIR"
    cd "$PERCH_MODEL_DIR"
    kaggle models instances versions download \
        google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1 \
        --untar
    echo "  Model downloaded to: $PERCH_MODEL_DIR"
    ls -la "$PERCH_MODEL_DIR"
else
    echo "perch_v2_cpu model already exists at: $PERCH_MODEL_DIR"
fi
echo ""

# ──────────────────────────────────────────────────────────────
# Prepare output directories (shared storage)
# ──────────────────────────────────────────────────────────────
cd "$EXPERIMENT_DIR"
mkdir -p logs artifacts

SHARED_ROOT="/storage/project/ps-clef2026_birdclef-0/shared"
mkdir -p "$SHARED_ROOT/embeddings/birdclef-2026/perch-v2/labeled"
mkdir -p "$SHARED_ROOT/embeddings/birdclef-2026/perch-v2/unlabeled"
mkdir -p "$SHARED_ROOT/artifacts/birdclef-2026/perch-ensemble-v1"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ──────────────────────────────────────────────────────────────
# Step 1: Build species mapping (re-run with CPU model labels)
# ──────────────────────────────────────────────────────────────
echo "=============================================="
echo "Step 1: Build species mapping"
echo "=============================================="
python scripts/build_species_mapping.py --config "$CONFIG"
echo ""

# ──────────────────────────────────────────────────────────────
# Step 2: Extract Perch embeddings + logits (CPU)
# ──────────────────────────────────────────────────────────────
echo "=============================================="
echo "Step 2: Extract Perch embeddings (${SPLIT} split)"
echo "=============================================="
python scripts/extract_perch_embeddings.py --config "$CONFIG" --split "$SPLIT"
echo ""

# ──────────────────────────────────────────────────────────────
# Step 3: Train probes + build metadata priors (labeled only)
# ──────────────────────────────────────────────────────────────
if [ "$SPLIT" = "labeled" ]; then
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
else
    echo "=============================================="
    echo "Steps 3-4: Skipped (probe training only runs on labeled split)"
    echo "=============================================="
    echo ""
fi

echo "=============================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "=============================================="
