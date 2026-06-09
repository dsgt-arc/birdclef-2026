#!/bin/bash
# Shared CUDA venv for exp 301 (LoRA + DAPT). Mirrors exp 103's mamba-ssm setup
# (SSAMBA needs CUDA kernels) and adds transformers/peft for EAT/BEATs. Persistent
# on scratch. The shared `birdclef` encoder library is on PYTHONPATH; exp 300
# (the extract/probe hub) is pulled in editable via exp 301's path dependency.
#
# NOTE: `module load cuda` must stay in the sbatch BODY, not here.
#
# Usage: source sbatch/_setup-venv.sh

set -eu

repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir=$repo_dir/user/anthony/experiments/301-encoder-adaptation
shared_src=$repo_dir/user/anthony/src

scratch_dir=$(realpath ~/scratch/birdclef-2026)
VENV_DIR="$scratch_dir/.venv-301"
VENV_MARKER="$VENV_DIR/.setup-done"

if [ ! -f "$VENV_MARKER" ]; then
    echo "=== First run: building persistent venv at $VENV_DIR ==="
    uv venv "$VENV_DIR" --python 3.11
    source "$VENV_DIR/bin/activate"

    uv pip install setuptools torch torchaudio --index-url https://download.pytorch.org/whl/cu126
    uv pip install triton
    uv pip install mamba-ssm causal-conv1d          # SSAMBA GPU kernels
    uv pip install transformers peft                # EAT / BEATs
    uv pip install polars scipy soundfile pyyaml numpy tqdm scikit-learn luigi

    touch "$VENV_MARKER"
    echo "=== Persistent venv built ==="
else
    echo "=== Using existing venv at $VENV_DIR ==="
    source "$VENV_DIR/bin/activate"
fi

# Editable exp301 (pulls exp300 via path dep); shared encoders via PYTHONPATH.
uv pip install -e "$exp_dir"
export PYTHONPATH="$shared_src:${PYTHONPATH:-}"

cd "$exp_dir"
