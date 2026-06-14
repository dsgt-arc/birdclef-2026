#!/bin/bash
#SBATCH --job-name=300-fit-head
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 60 -q inferno
#SBATCH --output=logs/fit-head-%j.out

# Fit one deployable ridge head on ALL training embeddings (focal + soundscape)
# over the 234-species roster -> artifacts/head.npz. Streaming single pass over the
# exp 106 Perch cache; CPU-only, ~minutes.
#
# Usage:
#   sbatch sbatch/fit-head.sbatch                                  # perch, focal+soundscape, λ=1e4
#   SOURCES=soundscape LAMBDA=1e4 sbatch sbatch/fit-head.sbatch    # soundscape-only ablation

set -euo pipefail
mkdir -p logs artifacts

# Run THIS worktree's code: derive the repo root from the submit dir (the exp dir
# is repo/user/anthony/experiments/300-...), overriding the ~/arc default so the new
# deploy/ module is picked up even when ~/arc points at the exp/301 worktree.
export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
echo "REPO_DIR=$REPO_DIR"

source sbatch/_setup-venv.sh

birdclef-300 fit-head \
    --features-dir "${FEATURES_DIR:-$HOME/scratch/birdclef-2026/features/exp106}" \
    --mapping "${MAPPING:-artifacts/perch_species_mapping.json}" \
    --sources "${SOURCES:-focal,soundscape}" \
    --lambda "${LAMBDA:-10000}" \
    --encoder "${ENCODER:-perch}" \
    --out "${OUT:-artifacts/head.npz}"
