# Experiment 015: Results

- **Status**: COMPLETE — NEGATIVE RESULT
- **Date**: 2026-03-08

## Summary

Adding a per-frame SED head with dual loss (clip + frame) to the frozen EfficientNet-B1
backbone did **not** improve over clip-level prediction. Final val/cmap of 0.237 is
essentially identical to exp 014 (soft labels without SED) and well below the exp 012
hard-label baseline (0.336).

## Results

| Model | val/cmap | val/auroc | train/cmap | train/loss |
|-------|----------|-----------|------------|------------|
| Exp 015: SED + soft labels | 0.237 | 0.812 | 0.266 | 0.019 |
| Exp 014: clip + soft labels | 0.238 | 0.804 | 0.263 | — |
| *Exp 012: clip + hard labels* | *0.336* | *—* | *—* | *—* |

- Job 4624410: 30 epochs, ~11 hours on RTX 6000
- Both clip_loss and frame_loss decreased (success criterion 3 met)
- But neither val/cmap nor val/auroc improved over baseline (criteria 1-2 failed)

## Analysis

### Why SED didn't help

1. **Same bottleneck as exp 014**: The dominant issue is soft label quality, not the
   head architecture. Both exp 014 (clip-level) and exp 015 (frame-level) converge
   to the same ~0.237 val/cmap, indicating the ceiling is set by the labels, not the
   pooling strategy.

2. **Frozen backbone limits temporal learning**: The SED head can only rearrange
   frozen features — it cannot learn new temporal representations. The attention
   pooling + framewise max pattern from top Kaggle solutions works because those
   models have trainable backbones that co-adapt with the SED head.

3. **1s resolution may be too coarse**: The 20→5 frame pooling matches soft label
   resolution (1s), but bird calls have sub-second temporal structure. With a frozen
   backbone producing 250ms-resolution features, pooling to 1s loses information.

4. **Train/infer divergence untested**: The attention (train) / framewise max (infer)
   pattern was implemented but may not matter with a frozen backbone where attention
   weights have limited capacity to specialize.

### Relationship to exp 014

Exp 014 and 015 form a controlled comparison:
- **Exp 014**: clip-level prediction with soft labels → 0.238
- **Exp 015**: frame-level SED with same soft labels → 0.237

The difference is negligible. This confirms the bottleneck is the soft labels
themselves, not whether supervision is applied at clip or frame level.

### What would need to change

For SED to provide value, it likely requires:
1. **Trainable backbone** (LoRA or full fine-tuning) so temporal features can co-adapt
2. **Hard labels** for focal data (don't replace ground truth with teacher estimates)
3. **Soft labels only for soundscape/unlabeled data** where no ground truth exists
4. **Higher temporal resolution** labels (sub-second) to exploit the backbone's 250ms output

## Conclusion

**SED architecture is not the bottleneck** — label quality is. The frame-level supervision
from teacher soft labels provides no advantage over clip-level supervision when the backbone
is frozen. Future SED experiments should use hard labels for focal data and reserve soft
labels for unlabeled data augmentation.
