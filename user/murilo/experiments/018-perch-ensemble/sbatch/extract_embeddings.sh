#!/bin/bash
#SBATCH -J exp018-perch-extract
#SBATCH -A paceship-clef2026_birdclef
#SBATCH -p gpu-rtx6000
#SBATCH -q inferno
#SBATCH --gres=gpu:1
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
# Runs all offline preparation steps:
#   1. Build species mapping (Perch -> BirdCLEF 2026)
#   2. Extract Perch embeddings + logits from soundscapes (GPU-accelerated TF)
#   3. Train LogReg probes + metadata priors
#   4. Verify and package artifacts for Kaggle
#
# All paths are read from config/perch_ensemble_v1.yaml.
# Embeddings → shared/embeddings/birdclef-2026/perch-v2/labeled/
# Probes     → shared/artifacts/birdclef-2026/perch-ensemble-v1/
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
nvidia-smi
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

echo "  Loading CUDA module..."
module load cuda/12.6.1 cudnn/9.2.0.82-12-cuda
echo "  CUDA: $(nvcc --version 2>/dev/null | tail -1 || echo 'nvcc not found')"

echo "  Installing experiment dependencies..."
uv pip install -e "$EXPERIMENT_DIR"

echo "  Installing TensorFlow (uses system CUDA via module)..."
uv pip install "tensorflow[and-cuda]"

# Point TF to the CUDA libs from the loaded module + pip-installed nvidia packages
NVIDIA_SITE_PACKAGES="$JOB_VENV/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_SITE_PACKAGES}/cublas/lib:${NVIDIA_SITE_PACKAGES}/cuda_cupti/lib:${NVIDIA_SITE_PACKAGES}/cuda_nvrtc/lib:${NVIDIA_SITE_PACKAGES}/cuda_runtime/lib:${NVIDIA_SITE_PACKAGES}/cudnn/lib:${NVIDIA_SITE_PACKAGES}/cufft/lib:${NVIDIA_SITE_PACKAGES}/cusolver/lib:${NVIDIA_SITE_PACKAGES}/cusparse/lib:${NVIDIA_SITE_PACKAGES}/nccl/lib:${NVIDIA_SITE_PACKAGES}/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

echo "  Virtual environment ready!"
echo ""

python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); gpus = tf.config.list_physical_devices('GPU'); print(f'GPU devices: {gpus}'); assert len(gpus) > 0, 'No GPU detected!'"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
echo ""

# ──────────────────────────────────────────────────────────────
# Prepare output directories (shared storage)
# ──────────────────────────────────────────────────────────────
cd "$EXPERIMENT_DIR"
mkdir -p logs artifacts

# Shared storage directories
SHARED_ROOT="/storage/project/ps-clef2026_birdclef-0/shared"
mkdir -p "$SHARED_ROOT/embeddings/birdclef-2026/perch-v2/labeled"
mkdir -p "$SHARED_ROOT/embeddings/birdclef-2026/perch-v2/unlabeled"
mkdir -p "$SHARED_ROOT/artifacts/birdclef-2026/perch-ensemble-v1"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ──────────────────────────────────────────────────────────────
# Step 1: Build species mapping (skip if already exists)
# ──────────────────────────────────────────────────────────────
MAPPING_FILE="${EXPERIMENT_DIR}/artifacts/perch_species_mapping.json"
if [ -f "$MAPPING_FILE" ]; then
    echo "=============================================="
    echo "Step 1: Species mapping already exists, skipping"
    echo "  $MAPPING_FILE"
    echo "=============================================="
else
    echo "=============================================="
    echo "Step 1: Build species mapping"
    echo "=============================================="
    python scripts/build_species_mapping.py --config "$CONFIG"
fi
echo ""

# ──────────────────────────────────────────────────────────────
# Step 2: Extract Perch embeddings + logits (GPU-accelerated)
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
