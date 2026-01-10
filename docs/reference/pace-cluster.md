# PACE Cluster Reference

Reference for Georgia Tech PACE cluster resources. This is the canonical source for PACE-specific facts used by other documentation.

## Configuration

Before using PACE, set your account in your environment:

```bash
# Add to ~/.bashrc
export PACE_ACCOUNT="pace-your_allocation_here"  # e.g., pace-dsgt_clef2025
```

All examples below use `${PACE_ACCOUNT}` - substitute your allocation.

## Disk Usage

### Home Directory (~/)
- **Quota**: 20 GB
- **Use for**: Code, configs, small files
- **Avoid**: Datasets, checkpoints, virtual environments, pip/uv caches

### Scratch Directory (~/scratch/)
- **Quota**: 15 TB
- **Use for**: Datasets, model checkpoints, virtual environments, experiment outputs
- **Path**: `~/scratch/birdclef-2026/`
- **Note**: Files may be purged after 60 days of inactivity

### Project Directory (shared storage)
- **Path**: `/storage/coda1/p-<allocation>/0/shared/` (e.g., `p-dsgt_clef2025`)
- **Use for**: Shared datasets, final model artifacts
- **Note**: Persistent storage for team collaboration

## SLURM Patterns

### Queue Options

| Queue | Time Limit | Use Case |
|-------|------------|----------|
| `embers` | 2 hours | Quick jobs, downloads, data prep |
| `inferno` | 12+ hours | Training runs |

### Resource Requests

```bash
# CPU job (data download, preprocessing)
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=4G

# Single GPU job (training, inference)
#SBATCH --nodes=1 --gres=gpu:1 --cpus-per-task=6 --mem-per-gpu=32G

# Specific GPU type
#SBATCH --constraint=RTX6000
#SBATCH --partition=gpu-l40s --gres=gpu:L40s
```

### Standard sbatch Header

```bash
#!/bin/bash
#SBATCH --job-name=my-job
#SBATCH --account=${PACE_ACCOUNT}
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=4G
#SBATCH -t60 -qembers
#SBATCH --output=logs/%j.out
#SBATCH --mail-type=END,FAIL --mail-user=your@email.edu
set -eu

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

# Environment setup
export UV_PROJECT_ENVIRONMENT=~/scratch/birdclef-2026/.venv
export HF_HOME=~/scratch/birdclef-2026/.cache/huggingface

hostname
date

# Your commands here
uv run python script.py
```

## Troubleshooting

### "Disk quota exceeded"
- Check quotas: `pace-quota`
- Move large files to scratch
- Clear caches: `rm -rf ~/.cache/huggingface ~/.cache/torch`
- Ensure `UV_PROJECT_ENVIRONMENT` points to scratch

### "No module named X" in sbatch
- Ensure virtual environment is activated or use `uv run`
- Check `UV_PROJECT_ENVIRONMENT` is set correctly
- Verify package is installed: `uv pip list | grep X`

### Dynamic library loading errors
- PyTorch/TensorFlow load `.so` files from venv at runtime
- Solution: Install venv to scratch (not home) where there's space
- Set `LD_LIBRARY_PATH` if needed

### Jobs stuck in queue
- Check cluster status: `pace-check-queue`
- Try different queue or reduce resource request
- Check account allocation: `pace-mybalance`
