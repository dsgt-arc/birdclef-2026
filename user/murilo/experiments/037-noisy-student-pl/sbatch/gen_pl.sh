#!/bin/bash
#SBATCH --job-name=exp037-gen-pl
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -qinferno
#SBATCH --output=logs/slurm_gen_pl_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Exp 037 — generate soft-PL parquets from the 0.929 ENSEMBLE teacher (pure transform over
# the two cached caches; NO model inference). Generates the round-1 grid variants: raw + t2
# (g05 is the banned over-inflating control — generate only if explicitly diagnosing).
#
# This is light enough (~2-3 min, pure pandas) to run on a login node with the interactive
# venv instead of SLURM:
#   ~/clef/birdclef-2026/.venv/bin/python scripts/generate_ensemble_pl.py --variant raw
#   ~/clef/birdclef-2026/.venv/bin/python scripts/generate_ensemble_pl.py --variant t2
#
# Or self-submit to a CPU node:
#   cd user/murilo/experiments/037-noisy-student-pl && bash sbatch/gen_pl.sh
set -eu

# Self-submit if not already under SLURM.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
    mkdir -p "$EXPERIMENT_DIR/logs"
    cd "$EXPERIMENT_DIR"
    echo "Submitting PL-generation job..."
    sbatch "$SCRIPT_DIR/gen_pl.sh"
    exit 0
fi

EXPERIMENT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$EXPERIMENT_DIR"
echo "Node: $(hostname) | Job: $SLURM_JOB_ID | $(date)"

export UV_CACHE_DIR="$TMPDIR/uv-cache"
JOB_VENV="$TMPDIR/birdclef-$SLURM_JOB_ID/.venv"
mkdir -p "$(dirname "$JOB_VENV")"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"
# Pure-pandas transform — no torch needed.
uv pip install numpy pandas pyarrow

for variant in raw t2; do
    echo "=== generating variant: $variant ==="
    srun python scripts/generate_ensemble_pl.py --variant "$variant"
done

echo "Done. PL parquets under shared/pseudo-labels/birdclef-2026/v037-ens-{raw,t2}/"
