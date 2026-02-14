# Experiment 009: BirdCLEF Pipeline

## Quick Reference

Package: `birdclef_009` (installed via workspace)
CLI commands: `birdclef-explore`, `birdclef-pretrain`, `birdclef-finetune`, `birdclef-predict`

## Running

All computation runs via sbatch. Never run training/inference on login nodes.

```bash
# Data setup
sbatch sbatch/download-data.sbatch
sbatch sbatch/explore-data.sbatch

# Pretraining
sbatch sbatch/pretrain-fast-dev.sbatch   # Sanity check
sbatch sbatch/pretrain.sbatch            # Full run

# Fine-tuning
sbatch sbatch/finetune-fast-dev.sbatch   # Sanity check
sbatch sbatch/finetune.sbatch            # Full run

# Inference
sbatch sbatch/predict.sbatch

# Tests
sbatch sbatch/test.sbatch
```

## Data Paths

- BirdCLEF 2025: `~/scratch/birdclef-2026/data/birdclef-2025/`
- Pretrain checkpoints: `logs/pretrain/`
- Finetune checkpoints: `logs/finetune/`

## Package Structure

- `birdclef_009.data.*` — data loading (pretrain, finetune, audio utils)
- `birdclef_009.models.*` — model architectures (pretrain MSPM, finetune SED)
- `birdclef_009.training.*` — Lightning modules
- `birdclef_009.inference.*` — prediction and submission formatting
- `birdclef_009.utils.*` — losses, augmentation, weight loading, GPU monitor
- `birdclef_009.cli.*` — CLI entry points

## Testing

Tests run on CPU by default. GPU tests are skipped without CUDA.
```bash
uv run pytest tests/ -v
```
