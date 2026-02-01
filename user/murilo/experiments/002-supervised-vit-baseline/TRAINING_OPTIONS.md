# Training Options for Experiment 002

## PACE Cluster Constraints

**Time Limit**: 72 hours (3 days) on gpu-rtx6000 partition

**Training Time Estimates** (XCM, ~160k samples, 50 epochs):

| Configuration | Est. Time | Fits in 24h? | Fits in 72h? |
|--------------|-----------|--------------|--------------|
| 1x RTX 6000 | 24-36 hours | Maybe | Yes |
| 2x RTX 6000 | 12-18 hours | Yes | Yes |
| 4x RTX 6000 | 6-9 hours | Yes | Yes |

**Recommendation**: Use 2 GPUs for XCM training (comfortable margin, ~2x speedup).

## Single GPU Training (Default)

**When to use**: HSN dataset (~5k samples), quick experiments, debugging

```bash
sbatch sbatch/train.sh
```

**Time**: HSN ~2-4 hours, XCM ~24-36 hours
**Pros**: Simple, no DDP complexity
**Cons**: Might hit 24h limit on XCM

## Multi-GPU Training (Recommended for XCM)

**When to use**: Full XCM training, need faster iteration

```bash
# Edit train_multigpu.sh to set --gres=gpu:2 (or 4)
sbatch sbatch/train_multigpu.sh
```

**Time**: ~12-18 hours with 2 GPUs, ~6-9 hours with 4 GPUs
**Speedup**: ~1.8-2x with 2 GPUs (not perfect 2x due to DDP overhead)

### How Multi-GPU Works

PyTorch Lightning handles all DDP (DistributedDataParallel) setup automatically:
- Model replicated across GPUs
- Gradients synchronized after each batch
- Effective batch size: `batch_size × num_gpus`

No code changes needed - just pass `--devices N --strategy ddp`:

```bash
python train.py --config config.yaml --devices 2 --strategy ddp
```

## Checkpoint Resumption

If training might exceed time limits, enable checkpoint saving:

**config.yaml**:
```yaml
callbacks:
  model_checkpoint:
    save_last: true
    every_n_epochs: 5
```

**Resume from checkpoint**:
```bash
python train.py --resume-from-checkpoint outputs/checkpoints/last.ckpt
```

Useful for: Training in multiple 24-hour chunks, recovering from timeouts.

## Gradient Accumulation

Simulate larger batch sizes without more VRAM:

**config.yaml**:
```yaml
training:
  batch_size: 16
  accumulate_grad_batches: 4  # Effective batch_size = 64
```

Useful for: OOM issues, limited memory. Note: Doesn't reduce training time.

## Troubleshooting

**Out of Memory (OOM)**:
1. Reduce batch_size to 16
2. Enable gradient accumulation
3. Verify precision: "16-mixed" (already default)

**Training Too Slow**:
1. Use 2-4 GPUs
2. Increase num_workers to 12-16
3. Check GPU utilization: `nvidia-smi` (target >80%)

**Job Killed at Time Limit**:
1. Use more GPUs (faster training)
2. Enable checkpoint resumption
3. Reduce max_epochs

## Monitoring

**GPU Utilization**:
```bash
ssh <node> nvidia-smi
# Target: >80% on all GPUs
# If <50%: Increase num_workers
```

**Training Progress**:
```bash
tail -f outputs/slurm/exp002-*.out
# Or check W&B dashboard
```

## Recommendations by Dataset

**HSN** (~5k samples, 21 species):
- Use 1 GPU (2-4 hours total)
- Command: `sbatch sbatch/train.sh`

**XCM** (~160k samples, 9,736 species):
- Use 2 GPUs (12-18 hours total)
- Command: `sbatch sbatch/train_multigpu.sh`

**Audio-MAE** (Experiment 003, 100 epochs):
- Use 2-4 GPUs (24-36 hours with 2 GPUs)
- Consider reducing to 50 epochs initially

## Quick Reference

```bash
# Test on HSN (single GPU)
sbatch sbatch/train.sh

# Full XCM training (2 GPUs)
sbatch sbatch/train_multigpu.sh

# Resume from checkpoint
python train.py --resume-from-checkpoint outputs/checkpoints/last.ckpt

# Custom GPU count
python train.py --config config.yaml --devices 4 --strategy ddp
```

**Note**: GPU-hours cost is similar (1 GPU × 24h ≈ 2 GPUs × 12h), but multi-GPU provides faster iteration and safety margin under time limits.
