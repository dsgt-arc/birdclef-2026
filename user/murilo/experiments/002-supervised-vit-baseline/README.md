# Experiment 002: Supervised ViT Baseline

**Status**: Ready for execution (infrastructure implemented)

## Overview

Build a robust PyTorch Lightning training pipeline for BirdSet with end-to-end supervised Vision Transformer training. This experiment validates infrastructure and provides a baseline for comparing Audio-MAE pre-training (Experiment 003).

## Quick Start

### 1. Install Dependencies

```bash
cd user/murilo
uv sync  # Install all dependencies including timm, wandb, einops
```

### 2. Quick Validation (Fast Dev Run)

Test the pipeline with 5 batches on HSN:

```bash
cd experiments/002-supervised-vit-baseline
python train.py --fast-dev-run
```

This should:
- Load HSN dataset (21 species, ~5,460 samples)
- Generate mel-spectrograms on-the-fly
- Run 5 batches of training and validation
- Complete in ~2-5 minutes

### 3. HSN Training (Infrastructure Validation)

Train on HSN for full 50 epochs:

```bash
python train.py --config config.yaml
```

Expected:
- GPU utilization >80%
- Training time: 2-4 hours
- Final validation cmAP > 0.30

### 4. Scale to XCM (Production)

Update `config.yaml`:
```yaml
data:
  dataset_name: "XCM"  # Change from HSN
```

Submit SLURM job:
```bash
cd sbatch
sbatch train.sbatch
```

Monitor progress:
```bash
tail -f ../outputs/slurm/exp002-vit-baseline-*.out
```

## Files

| File | Purpose |
|------|---------|
| `proposal.md` | Hypothesis, method, success criteria |
| `design.md` | Technical decisions and architecture |
| `tasks.md` | Implementation checklist |
| `config.yaml` | Hyperparameters and settings |
| `train.py` | Training script |
| `sbatch/train.sbatch` | SLURM batch script |

## Architecture

**Model**: ViT-Base with 1-channel patch embedding (modified from 3-channel RGB)
**Input**: Mel-spectrograms (128 mels, 500 frames) generated on-the-fly
**Task**: Multi-label bird species classification
**Loss**: BCEWithLogitsLoss
**Metrics**: cmAP (primary), F1-macro/micro, Top-1/5 accuracy

## Key Features

- **On-the-fly mel-spectrogram generation**: No need to store preprocessed data (~500GB savings)
- **Optimized data loading**: 8 workers, persistent workers, prefetching for GPU saturation
- **PyTorch Lightning**: Clean abstractions for training loop, metrics, callbacks
- **W&B integration**: Real-time monitoring of training curves and metrics
- **HSN → XCM workflow**: Fast iteration on small dataset, scale to production

## Troubleshooting

### GPU OOM
```bash
# Reduce batch size in config.yaml
data:
  batch_size: 16  # Try 16 instead of 32
```

### Low GPU Utilization
```bash
# Increase workers or prefetch factor
data:
  num_workers: 12  # Try 12 instead of 8
  prefetch_factor: 8  # Try 8 instead of 4
```

### DataLoader Too Slow
```bash
# Run with fewer workers for debugging
python train.py --num-workers 0  # Single-threaded, easier to debug
```

## Next Steps

After successful completion:
1. Document results in `results.md`
2. Analyze GPU utilization and data loading overhead
3. Compare HSN vs XCM performance
4. Use trained model as baseline for Audio-MAE (Experiment 003)

## References

- **BirdSet**: https://github.com/DBD-research-group/BirdSet
- **ViT**: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2021)
- **timm**: https://github.com/huggingface/pytorch-image-models
- **PyTorch Lightning**: https://lightning.ai/docs/pytorch/stable/
