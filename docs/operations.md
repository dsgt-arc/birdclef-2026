# Operations Guide

Operational patterns for running experiments on the Georgia Tech PACE cluster.

> **Prerequisites**: See [PACE Cluster Reference](reference/pace-cluster.md) for disk quotas, SLURM queues, and troubleshooting.

## Directory Structure

```
~/scratch/birdclef-2026/
├── .venv/                    # Python virtual environment (via uv)
├── checkpoints/              # Model checkpoints
│   ├── beats/
│   │   └── BEATs_iter3_plus_AS2M.pt
│   └── eat/
│       └── EAT-base_epoch30_finetune_AS2M/
├── data/                     # Downloaded datasets
│   ├── esc50/
│   └── audioset/
├── outputs/                  # Experiment outputs
│   └── 001-encoder-benchmark/
├── logs/                     # SLURM logs
└── .cache/                   # HuggingFace, pip, torch caches
    └── huggingface/
```

## Quick Start

```bash
# One-time setup (run on login node)
mkdir -p ~/scratch/birdclef-2026/{checkpoints,data,outputs,logs,.cache}

# Set environment (add to ~/.bashrc)
export PACE_ACCOUNT="pace-your_allocation"  # your SLURM account
export UV_PROJECT_ENVIRONMENT=~/scratch/birdclef-2026/.venv
export HF_HOME=~/scratch/birdclef-2026/.cache/huggingface
export TORCH_HOME=~/scratch/birdclef-2026/.cache/torch

# Install dependencies
cd ~/arc/birdclef-2026
uv venv ~/scratch/birdclef-2026/.venv
uv sync
```

## Using uv in sbatch Scripts

The recommended pattern is to use `uv run` which automatically handles the virtual environment:

```bash
#!/bin/bash
#SBATCH --job-name=my-job
#SBATCH --account=${PACE_ACCOUNT}
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=4G
#SBATCH -t60 -qembers
#SBATCH --output=logs/%j.out
set -eu

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

# Set environment to use scratch
export UV_PROJECT_ENVIRONMENT=~/scratch/birdclef-2026/.venv
export HF_HOME=~/scratch/birdclef-2026/.cache/huggingface
export TORCH_HOME=~/scratch/birdclef-2026/.cache/torch

# Run script with uv (handles venv activation automatically)
uv run python my_script.py
```

### Alternative: Manual Activation

For interactive work or complex scripts:

```bash
source ~/scratch/birdclef-2026/.venv/bin/activate
python my_script.py
```

## uv Workspace Pattern

For experiments with different dependencies, use uv workspaces. Each experiment can have its own `pyproject.toml`:

```
birdclef-2026/
├── pyproject.toml              # Root project
├── uv.lock                     # Locked dependencies
└── user/anthony/experiments/
    └── 001-encoder-benchmark/
        └── pyproject.toml      # Experiment-specific deps
```

Root `pyproject.toml`:
```toml
[tool.uv.workspace]
members = [
    "user/anthony/experiments/001-encoder-benchmark",
]
```

Then `uv run` from the experiment directory uses the right dependencies.

## Download Patterns

### Using sbatch for Downloads

Always use sbatch for downloads to avoid blocking the login node. Submit the download job from your experiment directory:

```bash
# Submit download job
sbatch user/anthony/experiments/001-encoder-benchmark/sbatch/download-checkpoints.sbatch

# Monitor job
squeue -u $USER
tail -f logs/<job_id>.out
```

### Download Tools

- **aria2c**: Fast parallel downloads (installed on PACE)
- **huggingface_hub**: For HuggingFace models/datasets
- **gdown**: For Google Drive links

Example aria2c for multiple files:
```bash
aria2c -c -j 3 -x 8 -i urls.txt -d ~/scratch/birdclef-2026/data/
```

### HuggingFace Downloads

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="worstchan/EAT-base_epoch30_finetune_AS2M",
    local_dir="~/scratch/birdclef-2026/checkpoints/eat/EAT-base",
    local_dir_use_symlinks=False
)
```

## Experiment Workflow

1. **Setup**: Run `sbatch sbatch/setup-scratch.sbatch` to create directory structure
2. **Download**: Run `sbatch sbatch/download-checkpoints.sbatch` for model weights
3. **Develop**: Edit code in home directory (`~/arc/birdclef-2026/`)
4. **Run**: Submit jobs that read from scratch, write outputs to scratch
5. **Analyze**: Results in `~/scratch/birdclef-2026/outputs/<experiment>/`

## Syncing Data

Set `PROJECT_DIR` to your project's shared storage path (see [PACE reference](reference/pace-cluster.md#project-directory-shared-storage)):

```bash
PROJECT_DIR=/storage/coda1/p-<allocation>/0/shared/birdclef-2026

# Scratch to project (for sharing)
rsync -av --progress ~/scratch/birdclef-2026/outputs/ $PROJECT_DIR/outputs/

# Project to scratch (for local access)
rsync -av --progress $PROJECT_DIR/data/ ~/scratch/birdclef-2026/data/
```

## Troubleshooting

See [PACE Cluster Reference](reference/pace-cluster.md#troubleshooting) for common issues.
