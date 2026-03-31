#!/bin/bash
# Shared venv setup for exp 103 sbatch scripts.
#
# Uses a PERSISTENT venv on scratch so mamba-ssm/causal-conv1d don't rebuild
# every job (~16 min saved). Editable packages (ssamba-hf, exp103) are
# re-installed each run to pick up code changes (fast, no compilation).
#
# Usage: source sbatch/_setup-venv.sh

set -eu

repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir=$repo_dir/user/anthony/experiments/103-ssamba-ssl
ssamba_dir=$repo_dir/user/anthony/experiments/004-ssamba-hf-port/ssamba_hf

scratch_dir=$(realpath ~/scratch/birdclef-2026)
VENV_DIR="$scratch_dir/.venv-103"
VENV_MARKER="$VENV_DIR/.setup-done"

module load cuda/12.6.1

if [ ! -f "$VENV_MARKER" ]; then
    echo "=== First run: building persistent venv at $VENV_DIR ==="
    uv venv "$VENV_DIR" --python 3.11
    source "$VENV_DIR/bin/activate"

    uv pip install setuptools torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu126
    uv pip install triton
    uv pip install mamba-ssm causal-conv1d

    touch "$VENV_MARKER"
    echo "=== Persistent venv built ==="
else
    echo "=== Using existing venv at $VENV_DIR ==="
    source "$VENV_DIR/bin/activate"
fi

# Always re-install editable packages to pick up code changes (fast, no build)
uv pip install -e "$ssamba_dir"
uv pip install -e "$exp_dir"

cd "$exp_dir"
