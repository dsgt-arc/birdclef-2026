#!/bin/bash
# Shared torch venv for exp 300 (BEATs / AST / EAT / SSAMBA-HF extraction + the
# ridge probe). Persistent on scratch so mamba-ssm/causal-conv1d don't rebuild
# every job. The shared `birdclef` encoder library is added via PYTHONPATH rather
# than installed, to avoid pulling its full (heavy) dependency closure.
#
# NOTE: `module load cuda` must stay in the sbatch BODY, not here (a sourced
# `module load` silently fails and torch falls back to CPU). See exp 107 CLAUDE.
#
# Usage: source sbatch/_setup-venv.sh

set -eu

repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir=$repo_dir/user/anthony/experiments/300-cross-encoder-transfer
shared_src=$repo_dir/user/anthony/src

scratch_dir=$(realpath ~/scratch/birdclef-2026)
VENV_DIR="$scratch_dir/.venv-300"
VENV_MARKER="$VENV_DIR/.setup-done"

if [ ! -f "$VENV_MARKER" ]; then
    echo "=== First run: building persistent venv at $VENV_DIR ==="
    uv venv "$VENV_DIR" --python 3.11
    source "$VENV_DIR/bin/activate"

    # CUDA torch first so the editable install below sees it as already satisfied.
    uv pip install setuptools torch torchaudio --index-url https://download.pytorch.org/whl/cu126
    uv pip install triton
    uv pip install mamba-ssm causal-conv1d          # SSAMBA GPU kernels
    uv pip install transformers peft                # EAT / AST (HF)
    uv pip install polars luigi scikit-learn scipy soundfile pyyaml numpy

    touch "$VENV_MARKER"
    echo "=== Persistent venv built ==="
else
    echo "=== Using existing venv at $VENV_DIR ==="
    source "$VENV_DIR/bin/activate"
fi

# Editable exp package (fast, no build); shared encoders via PYTHONPATH.
uv pip install -e "$exp_dir"
export PYTHONPATH="$shared_src:${PYTHONPATH:-}"

cd "$exp_dir"
