#!/bin/bash
# Shared venv setup for exp 107 sbatch scripts.
#
# Reuses the persistent exp 106 venv on scratch (torch + polars already
# installed) and layers einops + the exp 107 editable package on top. Stays
# off the home-directory quota per CLAUDE.md venv policy.
#
# Usage: source sbatch/_setup-venv.sh

set -eu

exp107_dir="$HOME/arc/birdclef-2026/user/anthony/experiments/107-hopfield-prototypes"
exp106_dir="$HOME/arc/birdclef-2026/user/anthony/experiments/106-perch-bayesian"

scratch_dir="$HOME/scratch/birdclef-2026"
VENV_DIR="$scratch_dir/.venv-106"
VENV_MARKER="$VENV_DIR/.setup-done-v1"
EXP107_MARKER="$VENV_DIR/.exp107-deps-v1"

if [ ! -f "$VENV_MARKER" ]; then
    echo "ERROR: expected exp 106 venv at $VENV_DIR (source exp106's _setup-venv.sh first)"
    exit 1
fi

echo "=== Using existing venv at $VENV_DIR ==="
source "$VENV_DIR/bin/activate"

if [ ! -f "$EXP107_MARKER" ]; then
    echo "=== Layering exp 107 deps (einops, pytest) ==="
    uv pip install einops pytest
    touch "$EXP107_MARKER"
fi

# Always re-install editable packages to pick up code changes.
echo "Installing editable package from $exp106_dir"
uv pip install -e "$exp106_dir"
echo "Installing editable package from $exp107_dir"
uv pip install -e "$exp107_dir"

cd "$exp107_dir"
