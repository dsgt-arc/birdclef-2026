# Experiment 014: Results

- **Status**: COMPLETE — NEGATIVE RESULT
- **Date**: 2026-03-08

## Summary

Soft labels from exp 013 fused-v2 significantly **degraded** performance compared to
the hard-label baseline (exp 012). Both conditions converged to ~0.238 val/cmap,
well below the 0.336 baseline. Adding soundscape data made no difference.

## Results

| Condition | val/cmap | val/auroc | train/cmap | Epochs | Kaggle Private |
|-----------|----------|-----------|------------|--------|----------------|
| A: Focal only | 0.238 | 0.804 | 0.263 | 30 | pending |
| B: Focal + soundscape | 0.238 | 0.801 | 0.258 | 30 | pending |
| *Exp 012 baseline* | *0.336* | *—* | *—* | *30* | *0.649* |

- Both conditions plateau early (~epoch 12) and show no further improvement
- Adding 117K soundscape chunks (+65% data) had zero effect
- Train/val gap is small (~0.025 cmAP), so overfitting is not the issue

## Jobs

| Condition | Job ID | Run Name | Checkpoint |
|-----------|--------|----------|------------|
| Focal | 4617047 | focal | `logs/focal/checkpoints/finetune-epoch=29-val/cmap=0.2383.ckpt` |
| Focal+soundscape | 4617048 | focal-soundscape | `logs/focal-soundscape/checkpoints/finetune-epoch=25-val/cmap=0.2377.ckpt` |

ONNX exports: Jobs 4645714 (focal), 4645715 (focal-soundscape)
Kaggle submissions: kernel v8 (focal), v9 (focal-soundscape)

## Analysis

### Why soft labels hurt

1. **Label quality**: The fused-v2 soft labels are noisy estimates from teacher models
   (BirdNET, Perch) with ~0.78-0.92 estimated accuracy. The hard labels from train.csv
   are ground truth. Replacing ground truth with noisy estimates is a net loss of
   information for focal recordings where the hard label is correct.

2. **Averaging destroys signal**: 1s soft labels averaged to 5s chunks dilute the
   supervision. A recording where the bird calls for 2 of 5 seconds gets label ~0.4
   instead of 1.0. While this is "more accurate" temporally, it gives the model weaker
   gradients for learning species features.

3. **Soundscape labels are too noisy**: Median confidence of soundscape soft labels is
   0.83 (vs 0.93 for focal). These are not reliable enough to improve training — they
   add volume without adding signal.

4. **Frozen backbone limits benefit**: With a frozen backbone, the model only trains a
   linear head. Soft labels could help more with a trainable backbone where temporal
   patterns in labels guide feature learning. For a linear probe, clean hard labels
   are strictly better.

### When soft labels might help

- With a trainable (unfrozen) backbone that can learn temporal patterns
- With per-frame SED supervision (tested in exp 015 — also negative)
- With higher-quality teacher labels (e.g., from a stronger ensemble)
- As pseudo-labels for unlabeled data only (not replacing ground truth)

## Conclusion

**Do not replace ground-truth hard labels with teacher soft labels for linear probe
training.** Soft labels should be used additively (for unlabeled data) rather than as
replacements for known-good labels.
