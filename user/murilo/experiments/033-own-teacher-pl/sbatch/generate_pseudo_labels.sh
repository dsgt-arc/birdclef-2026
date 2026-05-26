#!/bin/bash
#SBATCH --job-name=exp033-generate-pl
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_generate_pl_%j.log
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Exp 033 Phase 0.2: invoke exp 025's generate_pseudo_labels.py with our
# v2 config (T2 = distilled-sed-kd-5fold teacher, pure SED, no Perch tier).
# Output:
#   shared/pseudo-labels/birdclef-2026/v2-distilled-sed-kd-5fold/sed_pseudo_labels.parquet
#
# Walltime budget: 1hr (actual expected ~10-30 min — CPU work on pre-cached
# parquets, no GPU inference).
#
# Pre-requisite: sbatch/cache_inference.sh must complete first (writes the
# fold0..fold4 logits.parquet files this script reads from).
#
# Usage:
#   cd user/murilo/experiments/033-own-teacher-pl
#   sbatch sbatch/generate_pseudo_labels.sh

set -eu

PROJECT_NAME="${PROJECT_NAME:-ps-clef2026_birdclef-0}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    EXPERIMENT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
fi
USER_DIR="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
REPO_DIR="$(dirname "$(dirname "$USER_DIR")")"

# exp 025's generator
EXP025_DIR="${REPO_DIR}/user/murilo/experiments/025-iter-pseudo-labeling"

echo "=============================================="
echo "Exp 033 Phase 0.2: generate_pseudo_labels (v2 T2 parquet)"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          $(hostname)"
echo "Date:          $(date)"
echo "Git commit:    $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "Exp025 dir:    ${EXP025_DIR}"
echo "Config:        ${EXPERIMENT_DIR}/configs/pl_gen_v2.yaml"
echo ""

# Setup venv
echo "Setting up per-job virtual environment..."
export UV_CACHE_DIR="$TMPDIR/uv-cache"
JOB_VENV_DIR="$TMPDIR/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

# Install exp 025's package (provides generate_pseudo_labels.py)
uv pip install -e "$EXP025_DIR"

echo ""

cd "$EXP025_DIR"
mkdir -p "$EXPERIMENT_DIR/logs"

# Verify prerequisite: T2 inference cache must exist
T2_CACHE="/storage/project/${PROJECT_NAME}/shared/inference/birdclef-2026/distilled-sed-kd-5fold/train_soundscapes"
if [ ! -d "$T2_CACHE/fold0" ]; then
    echo "ERROR: T2 inference cache not found at $T2_CACHE/fold0/"
    echo "Run sbatch/cache_inference.sh first."
    exit 1
fi
echo "T2 cache prerequisite OK: $(ls $T2_CACHE | tr '\n' ' ')"

echo ""
echo "=============================================="
echo "Running generate_pseudo_labels.py..."
echo "=============================================="
srun python generate_pseudo_labels.py --config "${EXPERIMENT_DIR}/configs/pl_gen_v2.yaml"

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "=============================================="
