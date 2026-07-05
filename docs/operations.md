# Operations Guide

Operational patterns for running experiments on the Georgia Tech PACE cluster.

> **Prerequisites**: See [PACE Cluster Reference](references/pace-cluster.md) for disk quotas, SLURM queues, and troubleshooting.

## Virtual environment policy

Do **not** create `.venv` inside the repository or under `$HOME`. The home quota is small, and Python environments with Torch/audio stacks can exhaust it quickly.

Use job-local storage:

```bash
VENV_BASE="${TMPDIR:-/tmp}"
JOB_ID="${SLURM_JOB_ID:-$$}"
JOB_VENV_DIR="$VENV_BASE/birdclef-$JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"
uv pip install -e .
```

Rules:

- Never run bare `pip install`.
- Never run `pip install --user`.
- Never run `uv sync` in this repo; it may create `.venv` in the current working directory.
- Prefer `uv pip install ...` inside an activated temp venv.

## Data and artifact layout

Keep source code in the repo and large artifacts on scratch or shared project storage.

```text
$TMPDIR/birdclef-$SLURM_JOB_ID/     # job-local venv and temporary files
/storage/.../shared/birdclef-2026/  # shared datasets, checkpoints, outputs
user/<name>/experiments/NNN-name/   # experiment code, configs, docs
```

Common artifact directories:

```text
checkpoints/   # model weights
cache/         # HuggingFace, torch, audio caches
outputs/       # predictions, metrics, submissions
logs/          # SLURM logs
```

Set cache variables inside jobs so downloads do not land in `$HOME`:

```bash
export HF_HOME="$TMPDIR/hf"
export TORCH_HOME="$TMPDIR/torch"
export XDG_CACHE_HOME="$TMPDIR/cache"
```

## SLURM job skeleton

```bash
#!/bin/bash
#SBATCH --job-name=birdclef-job
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=4G
#SBATCH -t60
#SBATCH --output=logs/%j.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

VENV_BASE="${TMPDIR:?TMPDIR must be set on compute nodes}"
JOB_VENV_DIR="$VENV_BASE/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"
mkdir -p "$JOB_VENV_DIR"

uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"
uv pip install -e .

export HF_HOME="$TMPDIR/hf"
export TORCH_HOME="$TMPDIR/torch"
export XDG_CACHE_HOME="$TMPDIR/cache"

python my_script.py
```

For experiment-local packages, run `uv pip install -e .` from that experiment directory after activating the temp venv.

## Downloads

Use compute jobs for large downloads or preprocessing. Avoid long downloads on login nodes.

Useful tools:

- `aria2c` for parallel downloads
- `huggingface_hub` for model/data snapshots
- Kaggle CLI for competition data, where configured

Example:

```bash
aria2c -c -j 3 -x 8 -i urls.txt -d "$PROJECT_STORAGE/data"
```

## Syncing outputs

Set `PROJECT_STORAGE` to the shared project path for durable artifacts:

```bash
PROJECT_STORAGE=/storage/coda1/p-<allocation>/0/shared/birdclef-2026

# job scratch -> shared project storage
rsync -av --progress "$TMPDIR/outputs/" "$PROJECT_STORAGE/outputs/"

# shared project storage -> local scratch
rsync -av --progress "$PROJECT_STORAGE/data/" "$TMPDIR/data/"
```

## Troubleshooting

See [PACE Cluster Reference](references/pace-cluster.md#troubleshooting) for common issues.
