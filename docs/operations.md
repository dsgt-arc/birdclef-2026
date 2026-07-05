# Operations Guide

Current operational rules for running BirdCLEF jobs on PACE or similar SLURM clusters.

## Non-negotiable venv rule

Do **not** create `.venv` inside this repository or under `$HOME`. Home quota is small, and Torch/audio environments can exhaust it quickly.

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
- In SLURM jobs, also set `UV_CACHE_DIR="$TMPDIR/uv-cache"` so uv cache and venv are on the same filesystem.

## PACE storage and caches

| Location | Use |
|---|---|
| `$HOME` | code/config only; avoid caches and envs |
| `$TMPDIR` | job-local venvs, temporary files, copied working data |
| project/shared storage | durable datasets, checkpoints, model caches, outputs |

Set cache variables inside jobs so downloads do not land in `$HOME`:

```bash
export UV_CACHE_DIR="$TMPDIR/uv-cache"
export HF_HOME="$TMPDIR/hf"
export TORCH_HOME="$TMPDIR/torch"
export XDG_CACHE_HOME="$TMPDIR/cache"
```

For shared pretrained weights that should persist across jobs, prefer a project-storage cache instead of `$TMPDIR`.

## SLURM job skeleton

```bash
#!/bin/bash
#SBATCH --job-name=birdclef-job
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=4G
#SBATCH -t60
#SBATCH --output=logs/%j.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export UV_CACHE_DIR="$TMPDIR/uv-cache"
JOB_VENV="$TMPDIR/birdclef-$SLURM_JOB_ID/.venv"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"
uv pip install -e .

export HF_HOME="$TMPDIR/hf"
export TORCH_HOME="$TMPDIR/torch"
export XDG_CACHE_HOME="$TMPDIR/cache"

python my_script.py
```

For experiment-local packages, run `uv pip install -e .` from that experiment directory after activating the temp venv.

## Performance notes

- If data loading is a large fraction of step time, copy the working dataset/shards from network storage to `$TMPDIR` at job start.
- Keep durable model caches and outputs on project/shared storage; sync only final artifacts back from `$TMPDIR`.
- Prefer larger batch sizes over gradient accumulation when memory allows; accumulation saves memory but not compute.
- Use project/shared storage for large downloads or preprocessing outputs. Avoid long downloads on login nodes.

## Useful PACE commands

```bash
pace-quota         # quotas
pace-check-queue   # queue status
pace-mybalance     # allocation balance
```
