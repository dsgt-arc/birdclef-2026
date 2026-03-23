---
experiment: "015-student-training"
status: proposed
date: 2026-03-22
author: Murilo Gustineli
---

# Experiment 015: Student Training on Pseudo-Labeled Soundscapes

## Hypothesis

Training on focal recordings (hard labels) + labeled soundscapes (hard labels) +
pseudo-labeled unlabeled soundscapes (soft labels from exp 014) will improve LB
score by ≥0.02 over the exp 012 baseline (ProtoCLR 0.767, ConvNeXt 0.755, EAT 0.720).

## Motivation

Exp 012 showed that adding just 739 labeled soundscape segments improved all 3 models
by +0.022 to +0.060 LB. With 10,592 unlabeled soundscapes (~127K 5s segments), there
is far more soundscape data available — but it requires pseudo-labels.

The 2025 1st place winner used **iterative self-training** (3-4 rounds) where each
iteration produces better teachers for the next round. This experiment implements
the first iteration; subsequent rounds re-run exp 014 → exp 015.

### Iterative Self-Training Loop

```
Round 0: Exp 012 teachers (current best)
    → Exp 014: pseudo-label unlabeled data
    → Exp 015: train student models
    → Student becomes new teacher

Round 1: Exp 015 round-0 teachers
    → Exp 014: re-pseudo-label with improved teachers
    → Exp 015: retrain students
    → Repeat 2-3 times
```

## Approach

### Data Mix

| Source | Segments | Labels | Weight |
|--------|----------|--------|--------|
| Focal recordings | ~213K chunks | Hard (ground truth) | 1.0 |
| Labeled soundscapes | ~570 train segments | Hard (ground truth) | 1.0 |
| Pseudo-labeled soundscapes | ~127K segments | Soft (from exp 014) | Tunable (0.5–1.0) |

### Training Strategy

1. **Start from exp 012 checkpoint** (234-class head, soundscape-adapted)
2. **Hard labels for labeled data** — never replace ground truth
3. **Soft BCE loss for pseudo-labeled data** — teacher probabilities as targets
4. **Mixed batches**: Each batch samples from both labeled and pseudo-labeled data
5. **Lower LR** (1e-4 or lower) to preserve learned representations
6. **LoRA fine-tuning** (same as exp 012) — trainable backbone

### Loss Function

```
For labeled data:    BCE(pred, hard_label)         # standard binary cross-entropy
For pseudo-labeled:  BCE(pred, soft_label) × w     # soft targets, weighted
Total:               loss_labeled + w × loss_pseudo
```

Weight `w` controls the influence of pseudo-labeled data. Start with w=0.5,
sweep [0.25, 0.5, 0.75, 1.0].

### Variables

- **Independent**: Pseudo-label source (direct vs MixIT-enhanced from exp 014),
  pseudo-label weight `w`, confidence threshold, number of self-training iterations
- **Dependent**: LB score, per-species AUROC, focal test AUROC (forgetting check)
- **Controlled**: Same encoders, same LoRA config, same preprocessing as exp 012

## Evaluation

### Metrics
- **Primary**: Kaggle LB macro-AUROC
- **Secondary**: Offline test AUROC on labeled soundscapes, focal test AUROC
  (catastrophic forgetting check), per-taxon breakdown

### Baseline
- Exp 012 LB: ProtoCLR 0.767, ConvNeXt 0.755, EAT 0.720

### Success Criteria
- **Confirm if**: LB improves by ≥0.02 over exp 012 for at least one model
- **Reject if**: LB drops or focal test AUROC drops >0.02 (catastrophic forgetting)

## Dependencies

- Exp 014 pseudo-labels (proposed)
- Exp 012 checkpoints (complete)
- Exp 012 training infrastructure: `train.py`, `MixedDataModule`, SLURM scripts (complete)

## Leads to

- Iterative self-training (round 1+): retrained students become teachers for exp 014
- SED architecture experiments: if student models plateau, try SED head (Phase A from roadmap)
- Ensemble inference: submit best students as an ensemble (Phase C from roadmap)
