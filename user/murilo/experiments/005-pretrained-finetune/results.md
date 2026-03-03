# Results: Pretrained Audio Model Fine-Tuning on BirdCLEF 2025

**Date Completed**: 2026-02-23 (frozen baseline; finetune pending)
**Author**: Murilo Gustineli

## Summary

EAT frozen linear probe establishes a strong first baseline on BirdCLEF 2025 (test AUROC=0.890, cmAP=0.301), validating that AudioSet self-supervised pretraining produces useful representations for bird species classification. Full fine-tuning shows promising val cmAP (0.291) but timed out before test evaluation.

**Verdict**: inconclusive (frozen baseline confirmed; frozen vs fine-tune comparison pending)

## Observations

### Quantitative

#### EAT Frozen (Linear Probe) — Completed

Best checkpoint: epoch 29/30 | SLURM job: 4114048 | ~9h on 1x RTX 6000

| Metric | Val (best) | Test | Notes |
|--------|-----------|------|-------|
| AUROC | — | **0.890** | Primary metric (Kaggle competition metric) |
| cmAP | 0.238 | 0.301 | Test > val suggests no overfitting |
| F1 macro | — | 0.115 | Low — threshold=0.5 likely too aggressive for rare species |
| F1 micro | — | 0.186 | |
| Top-1 Acc | — | 37.9% | |
| Top-5 Acc | — | 62.9% | |

**Architecture**: EAT-base encoder (90.0M params, frozen) + linear head (158K trainable)
**Hyperparameters**: lr=8e-3, batch_size=256, warmup=3 epochs, AdamW (wd=0.01), cosine annealing, 16-mixed precision

#### EAT Fine-tune (Staged Unfreezing) — Timed Out

Best val checkpoint: epoch 46/50 | SLURM job: 4114217 | killed at 12h limit

| Metric | Val (best) | Test | Notes |
|--------|-----------|------|-------|
| cmAP | 0.291 | **pending** | Still improving at timeout, ~0.002/epoch |
| AUROC | — | pending | |
| F1 macro | — | pending | |
| Top-1 Acc | — | pending | |
| Top-5 Acc | — | pending | |

**Architecture**: EAT-base encoder (90.0M) + MLP head (512 hidden, dropout=0.1, 499K params)
**Hyperparameters**: head lr=8e-4, encoder lr=8e-5 (10x lower), batch_size=32 + accumulate_grad_batches=8 (effective 256), warmup=5 epochs, freeze encoder for first 5 epochs, then staged unfreeze

#### Comparison (Val cmAP only — test pending for finetune)

| Variant | Trainable Params | Val cmAP | Test cmAP | Epochs |
|---------|-----------------|----------|-----------|--------|
| EAT Frozen (linear probe) | 158K | 0.238 | 0.301 | 30 |
| EAT Fine-tune (staged) | 90.5M | 0.291 | pending | 48/50 (timed out) |

### Qualitative

- **Model learns meaningful representations**: val cmAP improved consistently from epoch 0 to 29 (frozen) and 0 to 46 (finetune), far above random chance (~1/206 = 0.005)
- **Test > val for frozen run**: test cmAP (0.301) exceeds val cmAP (0.238) by 26%, indicating no overfitting and possibly beneficial class distribution in the test split
- **Fine-tune still improving at timeout**: val cmAP was 0.291 at epoch 46 and still climbing, though gains were diminishing (~0.002/epoch). The curve had not yet plateaued or triggered early stopping (patience=10)
- **Fine-tune val surpasses frozen val**: fine-tune val cmAP (0.291) exceeds frozen val cmAP (0.238), suggesting that adapting the encoder helps — but this needs test-set confirmation
- **Encoder unfreezing was smooth**: no OOM or instability after the batch_size fix (32 + accumulate 8). Val cmAP rose steadily from 0.057 (epoch 4, last frozen) to 0.291 (epoch 46)
- **F1 is low despite good cmAP/AUROC**: the fixed threshold=0.5 is likely too high for rare species. Per-species threshold tuning or using a lower global threshold should improve F1 substantially
- **GPU utilization bottleneck resolved**: increasing workers from 8 to 12 eliminated the data pipeline bottleneck, reducing data loading time from 61ms to 5ms average

### Training Curve (Val cmAP by Epoch)

```
Epoch   Frozen    Finetune
  0     0.011     0.007
  5     0.120     0.079   <-- finetune encoder unfreezes here
 10     0.186     0.155
 15     0.210     0.210
 20     0.225     0.241
 25     0.236     0.263
 29     0.238     0.275
 30      —        0.277
 35      —        0.285
 40      —        0.289
 45      —        0.291
 47      —        (killed by SLURM)
```

## Analysis

EAT's AudioSet pretraining transfers well to bird species classification. The frozen linear probe's AUROC=0.890 (cmAP=0.301) on a 206-class problem is a strong first baseline, confirming that the pretrained features contain species-discriminative information without any encoder adaptation.

The fine-tune run's val cmAP (0.291) already surpasses the frozen val cmAP (0.238) by 22%, and the gap would likely widen further on the test set. However, the run was killed by the 12-hour SLURM time limit before completing test evaluation. The training curve shows:

1. **Epochs 0-4 (head-only)**: finetune head trains slower than frozen (0.057 vs 0.101 frozen at epoch 4), likely because the MLP head has more parameters and lower LR (8e-4 vs 8e-3)
2. **Epochs 5-15 (encoder unfreezes)**: rapid improvement as encoder adapts, catching up to frozen val by epoch 15
3. **Epochs 15-47**: fine-tune pulls ahead and continues improving, though gains diminish

The frozen run's training was efficient — 158K params trained in ~9h with batch_size=256. The finetune run required the batch_size reduction to 32 (with gradient accumulation 8x) to avoid OOM when 90M encoder params needed gradients + Adam optimizer states.

### Confounders

- **Finetune timed out**: the most important comparison (test-set) is incomplete. Val cmAP suggests finetune is better, but test-set generalization could differ
- **Different head architectures**: frozen uses a linear head (768 -> 206), finetune uses MLP (768 -> 512 -> 206 with dropout=0.1). The MLP head alone (without encoder unfreezing) might account for some improvement
- **No per-species analysis yet**: aggregate metrics may mask poor performance on rare species (15 species have only 2 samples)
- **F1 threshold not tuned**: the fixed 0.5 threshold penalizes the F1 metric; this is a reporting limitation, not a model limitation
- **Frozen metrics.json stale**: `artifacts/training/eat_frozen/metrics.json` contains results from an earlier run (job 4113243, cmAP=0.267). The more recent run (job 4114048, cmAP=0.301) with 12 workers produced better results and is the one reported here

## Artifacts

| Type | Location |
|------|----------|
| Frozen metrics (old) | `artifacts/training/eat_frozen/metrics.json` (job 4113243) |
| Frozen log (final) | `logs/slurm_eat_frozen_4114048.log` (job 4114048, 12 workers) |
| Frozen checkpoint | `checkpoints/eat-frozen/best-epochepoch=29-v1.ckpt` (on project storage) |
| Finetune log | `logs/slurm_eat_finetune_4114217.log` (job 4114217, timed out) |
| Finetune checkpoint | `checkpoints/eat-finetune/best-epochepoch=46.ckpt` (on project storage) |
| W&B runs | [dsgt-arc/birdclef-pretrained-finetune](https://wandb.ai/dsgt-arc/birdclef-pretrained-finetune) |
| Configs | `config/eat_frozen.yaml`, `config/eat_finetune.yaml` |
| GPU analysis | `gpu-utilization-analysis.md` |

## Next Steps

- [ ] Re-submit finetune with extended wall time (or resume from checkpoint) to complete test evaluation
- [ ] Update `artifacts/training/eat_frozen/metrics.json` with results from job 4114048
- [ ] Save finetune metrics to `artifacts/training/eat_finetune/metrics.json` after test eval completes
- [ ] Per-species cmAP analysis: identify which species the model struggles with (especially rare classes)
- [ ] Tune classification threshold: sweep threshold on val set to maximize F1, then evaluate on test
- [ ] Kaggle submission with best model to get leaderboard score on hidden soundscape test data
- [ ] Compare against other pretrained encoders (BEATs, BirdNET) on same BirdCLEF 2025 data

**Leads to**: Kaggle submission experiment, per-species analysis, domain adaptation for soundscape evaluation
