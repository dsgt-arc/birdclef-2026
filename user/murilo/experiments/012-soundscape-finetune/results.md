# Results: Labeled Soundscape Fine-Tuning

**Date Started**: 2026-03-21
**Author**: Murilo Gustineli

## Summary

Soundscape fine-tuning improves LB scores by expanding from 206 → 234 species and
adapting to the soundscape domain. ProtoCLR improved from 0.736 → **0.767** (+0.031).

## Kaggle Leaderboard Results

Competition metric: **macro-averaged ROC-AUC** (skips classes with no true positive labels).

### Exp 012: Soundscape Fine-Tuned Models (234 classes)

| Model | Training Data | Species | Public LB Score | vs Exp 011 | Date |
|-------|--------------|---------|----------------:|----------:|------|
| ProtoCLR LoRA attn | Focal + soundscape | 234/234 | **0.767** | +0.031 | 2026-03-21 |
| EAT LoRA attn+MLP head | Focal + soundscape | 234/234 | — | — | — |
| ConvNeXt LoRA MLP+MLP head | Focal + soundscape | 234/234 | — | — | — |

### Exp 011 Baseline (for comparison)

| Model | Training Data | Species | Public LB Score |
|-------|--------------|---------|----------------:|
| ProtoCLR fulldata | Focal only | 206/234 | **0.736** |
| EAT fulldata | Focal only | 206/234 | **0.698** |
| ConvNeXt fulldata | Focal only | 206/234 | **0.695** |

### Training Results (offline test split, 234 classes)

| Model | Best Epoch | Stopped | Test AUROC | Test cmAP |
|-------|-----------|---------|-----------|----------|
| ProtoCLR | 11 | 13 | 0.845 | 0.481 |
| EAT | 3 | 11 | 0.838 | 0.452 |
| ConvNeXt | — | — | — | — |

**Note**: Test AUROC (0.838–0.845) is not comparable to exp 011 (0.929–0.938) because
exp 012 evaluates on 234 classes including 28 barely-trained soundscape-only species,
while exp 011 evaluated on 206 focal species only.

## Observations

- **ProtoCLR soundscape: 0.767 LB (+0.031 over exp 011 fulldata)**: The 206→234
  species expansion and soundscape domain adaptation both contribute to the improvement.
  This is the best single-model score so far.

- **EAT converged very fast**: Best epoch 3 (vs epoch 17 on exp 011). The lower LR
  (1e-4 vs 3e-4) combined with the warm start from exp 011 means the model adapts quickly.

## Artifacts

| Type | Location |
|------|----------|
| Checkpoints | `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/012-soundscape-finetune/checkpoints/` |
| Metrics | `artifacts/training/{protoclr,eat,convnext}_soundscape/metrics.json` |
| Code | Branch `mg/012-soundscape-finetune`, PR #17 |
