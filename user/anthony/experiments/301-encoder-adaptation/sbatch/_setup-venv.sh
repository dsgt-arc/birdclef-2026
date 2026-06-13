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

    # Mirror exp 300's proven stack (this venv builds the same encoders via exp
    # 300's registry, so it needs the same dependency closure). Pin the matched
    # torch/torchaudio/torchvision trio — bare `timm` otherwise pulls a newer
    # torchvision that bumps torch (2.11->2.12) and de-syncs torchaudio.
    uv pip install setuptools "torch==2.11.0" "torchaudio==2.11.0" "torchvision==0.26.0" \
        --index-url https://download.pytorch.org/whl/cu126
    uv pip install triton
    # mamba-ssm/causal-conv1d build from source and import torch at build time;
    # uv's build isolation hides the torch above, so disable it and pre-seed the
    # build deps their setup.py imports.
    uv pip install ninja packaging setuptools wheel
    uv pip install mamba-ssm causal-conv1d --no-build-isolation   # SSAMBA GPU kernels
    # transformers<5: EAT's HF remote-code model targets the 4.x PreTrainedModel API.
    # timm: required by EAT's remote code. pandas: shared birdclef.__init__ eagerly
    # imports ESC50Dataset (pandas) whenever any encoder is imported.
    uv pip install "transformers<5" peft timm pandas
    uv pip install polars luigi scikit-learn scipy soundfile pyyaml numpy tqdm

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
