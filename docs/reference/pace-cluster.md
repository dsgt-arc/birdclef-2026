# PACE Cluster Reference

Quick reference for Georgia Tech PACE cluster resources.

> **For writing/reviewing sbatch scripts**: See `.claude/commands/SKILLS.md` for comprehensive patterns and checklists.

## Storage

| Location | Quota | Use For |
|----------|-------|---------|
| `~/` | 20 GB | Code, configs |
| `~/scratch/` | 15 TB | Datasets, venvs, outputs (60-day purge policy) |
| `/storage/coda1/p-<alloc>/0/shared/` | Varies | Team-shared data |
| `${TMPDIR}` | ~1 TB | Job-local temp (auto-cleaned) |

## GPU Partitions

| Partition | GPU | VRAM | Availability |
|-----------|-----|------|--------------|
| `gpu-rtx6000` | RTX 6000 | 24GB | **Best** (default) |
| `gpu-v100` | V100 | 16/32GB | Good |
| `gpu-l40s` | L40S | 48GB | Moderate |
| `gpu-a100` | A100 | 40/80GB | Limited |
| `gpu-h100` | H100 | 80GB | Very limited |

## Quick Commands

```bash
# Check quotas
pace-quota

# Check queue status
pace-check-queue

# Check account balance
pace-mybalance
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Disk quota exceeded | Move files to scratch, clear `~/.cache/torch` |
| No module found | Use `uv run`, check `UV_PROJECT_ENVIRONMENT` |
| Jobs stuck | Try different partition, reduce resources |
