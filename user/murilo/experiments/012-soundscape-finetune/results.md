# Results: Labeled Soundscape Fine-Tuning

**Date Started**: 2026-03-21
**Author**: Murilo Gustineli

## Summary

(Pending — results will be recorded after training completes.)

## Kaggle Leaderboard Results

Competition metric: **macro-averaged ROC-AUC** (skips classes with no true positive labels).

### Exp 012: Soundscape Fine-Tuned Models (234 classes)

| Model | Training Data | Species | Public LB Score | vs Exp 011 | Date |
|-------|--------------|---------|----------------:|----------:|------|
| ProtoCLR LoRA attn | Focal + soundscape | 234/234 | — | — | — |
| ConvNeXt LoRA MLP+MLP head | Focal + soundscape | 234/234 | — | — | — |
| EAT LoRA attn+MLP head | Focal + soundscape | 234/234 | — | — | — |

### Exp 011 Baseline (for comparison)

| Model | Training Data | Species | Public LB Score |
|-------|--------------|---------|----------------:|
| ProtoCLR fulldata | Focal only | 206/234 | **0.736** |
| EAT fulldata | Focal only | 206/234 | **0.698** |
| ConvNeXt fulldata | Focal only | 206/234 | **0.695** |

## Observations

(Pending)

## Artifacts

| Type | Location |
|------|----------|
| Checkpoints | `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/012-soundscape-finetune/checkpoints/` |
| Code | Branch `mg/012-soundscape-finetune` |
