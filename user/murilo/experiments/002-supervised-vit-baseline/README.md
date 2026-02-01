# Experiment 002: Supervised ViT Baseline

**Status**: Ready for execution

See `proposal.md` for hypothesis, method, and success criteria.
See `design.md` for technical decisions and architecture.

## Quick Start

```bash
# 1. Install dependencies
cd user/murilo && uv sync

# 2. Quick validation (5 batches, ~5 minutes)
cd experiments/002-supervised-vit-baseline
python train.py --fast-dev-run

# 3. Full HSN training (50 epochs, 2-4 hours)
sbatch sbatch/train.sh
```

## Troubleshooting

**GPU OOM**: Reduce `batch_size` to 16 in `config.yaml`

**Low GPU utilization**: Increase `num_workers` to 12 in `config.yaml`

**Monitor progress**:
```bash
tail -f outputs/slurm/exp002-*.out
```
