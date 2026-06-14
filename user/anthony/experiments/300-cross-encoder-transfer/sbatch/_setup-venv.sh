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

# REPO_DIR override lets a job run from a specific worktree (e.g. the exp/300
# worktree) instead of whatever ~/arc/birdclef-2026 points at (the exp/301 main
# worktree). Defaults to the historical path so extract/probe jobs are unchanged.
repo_dir=$(realpath "${REPO_DIR:-$HOME/arc/birdclef-2026}")
exp_dir=$repo_dir/user/anthony/experiments/300-cross-encoder-transfer
shared_src=$repo_dir/user/anthony/src

scratch_dir=$(realpath ~/scratch/birdclef-2026)
VENV_DIR="$scratch_dir/.venv-300"
VENV_MARKER="$VENV_DIR/.setup-done"

if [ ! -f "$VENV_MARKER" ]; then
    echo "=== First run: building persistent venv at $VENV_DIR ==="
    uv venv "$VENV_DIR" --python 3.11
    source "$VENV_DIR/bin/activate"

    # CUDA torch stack — pin the matched trio. timm/torchvision otherwise resolve
    # to a newer torchvision that bumps torch (2.11->2.12) and de-syncs torchaudio,
    # which breaks the transformers Wav2Vec2 import. (torch 2.11 <-> torchvision
    # 0.26 <-> torchaudio 2.11.)
    uv pip install setuptools "torch==2.11.0" "torchaudio==2.11.0" "torchvision==0.26.0" \
        --index-url https://download.pytorch.org/whl/cu126
    uv pip install triton
    # mamba-ssm/causal-conv1d build from source and import torch at build time.
    # uv's build isolation hides the torch installed above (ModuleNotFoundError:
    # torch), so disable isolation and pre-seed the build deps that their
    # setup.py imports (ninja/packaging/setuptools/wheel).
    uv pip install ninja packaging setuptools wheel
    uv pip install mamba-ssm causal-conv1d --no-build-isolation   # SSAMBA GPU kernels
    # transformers<5: EAT's HF remote-code model targets the 4.x PreTrainedModel
    # API (_tied_weights_keys); 5.x's all_tied_weights_keys breaks from_pretrained.
    # timm: required by EAT's remote code. pandas: the shared birdclef.__init__
    # eagerly imports ESC50Dataset (pandas) when any encoder is imported.
    uv pip install "transformers<5" peft timm pandas
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
