---
name: slurm
description: PACE cluster SLURM job management. Use for writing, reviewing, and submitting sbatch scripts. Covers GPU partitions (gpu-rtx6000 default), queues, storage hierarchy, and job monitoring.
---

# PACE Cluster SLURM Skills

Comprehensive guide for writing and reviewing SLURM sbatch scripts on Georgia Tech PACE cluster.

## PACE Cluster Overview

### Storage Hierarchy

| Location | Quota | Speed | Use For |
|----------|-------|-------|---------|
| `~/` (Home) | 20 GB | Slowest | Code, configs, small files |
| `~/scratch/` | 15 TB | Medium | Datasets, checkpoints, venvs, outputs |
| `/storage/coda1/p-<alloc>/0/shared/` | Varies | Medium | Team-shared datasets, final artifacts |
| `${TMPDIR}` (Node NVMe) | ~1 TB | **Fastest** | Temporary files during job execution |

**Important Notes:**
- Scratch files may be purged after 60 days of inactivity
- Always use `${TMPDIR}` for job-local temporary files (auto-cleaned after job)
- Install virtual environments to scratch, not home

### Account Configuration

All sbatch scripts should use the project account:
```bash
#SBATCH --account=paceship-dsgt_clef2026
```

## GPU Partitions

**Default Recommendation**: Use `gpu-rtx6000` for most jobs - best availability.

| Partition | GPU Type | VRAM | Availability | When to Use |
|-----------|----------|------|--------------|-------------|
| `gpu-rtx6000` | RTX 6000 | 24GB | **Best** | Default for most training |
| `gpu-v100` | V100 | 16/32GB | Good | Inference, small models |
| `gpu-l40s` | L40S | 48GB | Moderate | Large batch sizes |
| `gpu-a100` | A100 | 40/80GB | Limited | High-performance training |
| `gpu-h100` | H100 | 80GB | Very limited | Maximum performance |
| `gpu-h200` | H200 | 141GB | Rare | Extremely large models |

**Partition Request Examples:**
```bash
# Default (recommended)
#SBATCH -p gpu-rtx6000
#SBATCH --gres=gpu:1

# For large models needing more VRAM
#SBATCH -p gpu-l40s
#SBATCH --gres=gpu:L40s

# High-performance (if available)
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:A100
```

## Queue Options (QOS)

| Queue | Time Limit | Use Case |
|-------|------------|----------|
| `embers` | 6 hours | GPU jobs, training runs |
| `inferno` | 24 hours | CPU-only, long preprocessing |

## Quick Patterns

### Header and Metadata

```bash
#!/bin/bash
#SBATCH --job-name=descriptive-name
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-rtx6000
#SBATCH -t360 -qembers
#SBATCH --output=logs/job-name-%j.out
# Optional for array jobs:
#SBATCH --array=0-14

# Usage: sbatch --export=VAR1=value1,VAR2=value2 script.sbatch
#
# Description of what this job does
# - Key features
# - Expected runtime
# - Resource requirements
```

**Resource Guidelines:**
- Match `--cpus-per-task` to actual parallelism (DataLoader workers, copy threads)
- Use `--mem-per-cpu` (not `--mem`) for better scheduling
- Time limits: `-t` in minutes - 120 (2h), 360 (6h), 1440 (24h)

### Error Handling and Setup

```bash
set -eu  # Exit on error, undefined variables

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

# Configuration with defaults
STUDENT=${STUDENT:-convnext-small}
MAX_EPOCHS=${MAX_EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-64}

echo "=============================================="
echo "Job: descriptive-name"
echo "=============================================="
echo "Config:"
echo "  Student: ${STUDENT}"
echo "  Max epochs: ${MAX_EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
hostname
date
nvidia-smi  # For GPU jobs
```

### Environment Setup

```bash
scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)

export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv

cd $repo_dir
```

**Key points:**
- Use `scratch_dir` for large data, outputs, caches
- Use `repo_dir` for code
- Export environment variables before running Python

### Python Execution

**External script (recommended for complex jobs):**
```bash
uv run python scripts/train.py \
    --model "$STUDENT" \
    --epochs $MAX_EPOCHS \
    --batch-size $BATCH_SIZE \
    --output-dir $scratch_dir/outputs/exp003
```

**Key points:**
- Use `uv run` for dependency management
- Pass variables as arguments, not environment variables
- Use quotes for string variables that might have spaces

## Common Pitfalls

| Bad | Good | Why |
|-----|------|-----|
| `num_workers=4` (hardcoded) | `num_workers=8` (match CPUs) | Use available resources |
| `cd ~/arc/project` | `cd $repo_dir` | Portable paths |
| `cp $src/* $dst/` | `ls $src/* \| xargs -P 8 -I {} cp {} $dst/` | Parallel copy |
| `python train.py` | `uv run python train.py` | Managed environment |

## Review Checklist

When reviewing an sbatch script, verify:

- [ ] `set -eu` at the top for error handling
- [ ] Resource requests match actual usage (CPUs, memory, GPU)
- [ ] Clear usage comment with examples
- [ ] Echo job config at start (hostname, date, nvidia-smi)
- [ ] Use `${SLURM_SUBMIT_DIR}` or repo_dir variable
- [ ] `UV_PROJECT_ENVIRONMENT` exported and points to scratch
- [ ] Local NVMe staging for large data (`${TMPDIR}`)
- [ ] Parallel operations match CPU count
- [ ] Python execution uses `uv run`
- [ ] Idempotent operations (skip already-completed work)
- [ ] Error checking and summary at the end
- [ ] Logs saved to identifiable files (`logs/job-${SLURM_JOB_ID}.out`)

## Troubleshooting

### "Disk quota exceeded"
- Check quotas: `pace-quota`
- Move large files to scratch
- Clear caches: `rm -rf ~/.cache/torch`
- Ensure `UV_PROJECT_ENVIRONMENT` points to scratch

### "No module named X" in sbatch
- Use `uv run` for all Python commands
- Check `UV_PROJECT_ENVIRONMENT` is set correctly
- Verify package: `uv pip list | grep X`

### Dynamic library loading errors
- PyTorch/TensorFlow load `.so` files from venv at runtime
- Solution: Install venv to scratch (not home) where there's space

### Jobs stuck in queue
- Check cluster status: `pace-check-queue`
- Try different partition or reduce resource request
- Check account allocation: `pace-mybalance`

## See Also

For detailed patterns, complete examples, DataLoader configuration, array job patterns, and GPU monitoring setup, see [reference.md](reference.md).
