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
# Invoke exp 025's generate_pseudo_labels.py with one of our exp 033 configs.
# Defaults to pl_gen_v2 (T2 = distilled-sed-kd-5fold teacher, no filter).
# Override CONFIG to point at any sibling YAML (pl_gen_v3 = same T2 source
# but with Sydorskyi-style chunk-discard + per-class trim filters applied).
#
# Walltime budget: 1hr (actual expected ~5-15 min — CPU work on pre-cached
# parquets, no GPU inference).
#
# Pre-requisite: sbatch/cache_inference.sh must complete first (writes the
# fold0..fold4 logits.parquet files generate_pseudo_labels.py reads from).
#
# Usage:
#   cd user/murilo/experiments/033-own-teacher-pl
#
#   # Default (v2 — no filter, replicates the LB 0.886-0.899 runs):
#   sbatch sbatch/generate_pseudo_labels.sh
#
#   # v3 (Sydorskyi filters — chunk_discard + trim for Phase 5 Test 1):
#   CONFIG=configs/pl_gen_v3.yaml sbatch sbatch/generate_pseudo_labels.sh
#
#   # Any other custom config:
#   CONFIG=configs/my_variant.yaml sbatch sbatch/generate_pseudo_labels.sh

set -eu

# Default to v2 (Phase 0.2 historical); Phase 5 callers override to v3.
CONFIG="${CONFIG:-configs/pl_gen_v2.yaml}"
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
echo "Exp 033: generate_pseudo_labels"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          $(hostname)"
echo "Date:          $(date)"
echo "Git commit:    $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "Exp025 dir:    ${EXP025_DIR}"
echo "Config:        ${EXPERIMENT_DIR}/${CONFIG}"
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
# Resolve CONFIG relative to EXPERIMENT_DIR (so callers can pass either
# `configs/pl_gen_v3.yaml` or an absolute path).
if [[ "$CONFIG" = /* ]]; then
    CONFIG_PATH="$CONFIG"
else
    CONFIG_PATH="${EXPERIMENT_DIR}/${CONFIG}"
fi
echo "Resolved config path: ${CONFIG_PATH}"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: config not found at $CONFIG_PATH" >&2
    exit 1
fi
srun python generate_pseudo_labels.py --config "${CONFIG_PATH}"

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "=============================================="
