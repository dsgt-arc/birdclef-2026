# Experiment 011: LoRA Baseline on BirdCLEF 2026

Retrain proven LoRA fine-tuning pipelines (EAT, ConvNeXt, ProtoCLR) on BirdCLEF 2026 focal training data (35,549 recordings, ~206 species).

## Quick Start

```bash
cd user/murilo/experiments/011-lora-baseline-2026

# Fast dev run (pipeline validation)
bash sbatch/run.sh config/eat_lora_attn_mlphead.yaml --fast-dev-run

# GPU memory check
bash sbatch/run.sh config/eat_lora_attn_mlphead.yaml --gpu-check 2

# Submit all 3 configs for full training
bash sbatch/run.sh --all

# Dry run (preview sbatch commands)
bash sbatch/run.sh --all --dry-run
```

## Configs

| Config | Encoder | LoRA Targets | Batch Size | Effective BS |
|--------|---------|-------------|------------|-------------|
| `eat_lora_attn_mlphead.yaml` | EAT-base (768d) | qkv, proj | 32 | 256 |
| `convnext_lora_mlp_mlphead.yaml` | ConvNeXt-Base (1024d) | fc1, fc2 | 128 | 256 |
| `protoclr_lora_attn.yaml` | ProtoCLR/CvT-13 (384d) | attn.proj_q/k/v/proj | 256 | 256 |

All configs use: LoRA rank=8, alpha=16, dropout=0.1, lr=3e-4, 50 epochs max, early stopping patience=10.

## Changes from 2025 Experiments (006/007/009)

- **Data path**: Points to `/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026`
- **Unified train.py**: Single script handles all 3 encoders (was 3 separate scripts)
- **W&B project**: All runs log to `birdclef-2026-lora-finetune`
- **Hyperparameters**: Identical to 2025 for fair comparison

## Outputs

- Checkpoints: `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/011-lora-baseline-2026/checkpoints/`
- W&B: `dsgt-arc/birdclef-2026-lora-finetune`
- Test results: `outputs/test_results/`
- SLURM logs: `logs/`
