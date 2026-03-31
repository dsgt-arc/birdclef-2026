#!/bin/bash
# Shared venv setup for exp 104 sbatch scripts.
#
# Uses a PERSISTENT venv on scratch to avoid reinstalling torch each job.
# Editable packages re-installed each run to pick up code changes.
#
# Usage: source sbatch/_setup-venv.sh

set -eu

repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir=$repo_dir/user/anthony/experiments/104-wavtokenizer-retrieval
wavtok_dir=$repo_dir/vendor/WavTokenizer

scratch_dir=$(realpath ~/scratch/birdclef-2026)
VENV_DIR="$scratch_dir/.venv-104"
VENV_MARKER="$VENV_DIR/.setup-done"

module load cuda/12.6.1

if [ ! -f "$VENV_MARKER" ]; then
    echo "=== First run: building persistent venv at $VENV_DIR ==="
    uv venv "$VENV_DIR" --python 3.11
    source "$VENV_DIR/bin/activate"

    uv pip install setuptools torch torchaudio --index-url https://download.pytorch.org/whl/cu126
    # WavTokenizer inference deps
    uv pip install encodec vocos einops scipy

    touch "$VENV_MARKER"
    echo "=== Persistent venv built ==="
else
    echo "=== Using existing venv at $VENV_DIR ==="
    source "$VENV_DIR/bin/activate"
fi

# Always re-install editable exp104 to pick up code changes
uv pip install -e "$exp_dir"

# WavTokenizer vendor code on PYTHONPATH (not pip-installable)
export PYTHONPATH="$wavtok_dir:${PYTHONPATH:-}"

cd "$exp_dir"
