# Experiment 014: EfficientNet with Soft Labels

## Hypothesis

Training the frozen EfficientNet-B1 backbone (from exp 012) with temporally-localized
soft labels (from exp 013 fused-v2) will outperform hard recording-level labels, because:

1. Soft labels provide per-second species presence probabilities, so 5s chunks where the
   bird sings for 2s get ~0.4 label instead of 1.0, reducing label noise.
2. Soundscape recordings (564K tokens) become usable training data via soft labels,
   increasing training data by ~60%.

## Baseline

Exp 012 (frozen EfficientNet-B1, hard labels): **0.336 val/cmap**, **0.649 Kaggle private**

## Method

- **Architecture**: Identical to exp 012 — frozen EfficientNet-B1 backbone + linear head
- **Training labels**: Fused-v2 soft labels from exp 013 (1s resolution, averaged to 5s chunks)
- **Validation labels**: Hard labels from train.csv (same split as exp 012, seed=42)
- **Loss**: FocalBCELoss (gamma=2) — works with float targets in [0,1]
- **Data**: Focal only (default) or focal + soundscape (--include-soundscape flag)

## Conditions

| Condition | Training Data | Expected val/cmap |
|-----------|--------------|-------------------|
| A: Soft labels, focal only | ~27K recordings | 0.34-0.36 |
| B: Soft labels, focal + soundscape | ~37K recordings | 0.35-0.38 |

## Metrics

- Primary: val/cmap (class-mean average precision)
- Secondary: Kaggle private score
- Comparison: Must exceed exp 012 baseline (0.336 val/cmap)

## Dependencies

- Exp 009: FinetuneLightningModule, FocalBCELoss, SpecAugment, audio utils
- Exp 012: EfficientNetClassifier model architecture
- Exp 013: Fused-v2 soft labels at ~/scratch/birdclef-2026/outputs/exp013/fused-v2/
