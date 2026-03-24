---
experiment: "015-student-training"
status: in_progress
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

Weight `w` controls the influence of pseudo-labeled data. Round 0 uses w=0.5;
weight sweep [0.1, 0.25, 0.75, 1.0] deferred to future work.

### Temperature Scaling

The 2025 1st place winner used T=0.55-0.6 to sharpen pseudo-labels. A sweep
on our labeled validation (75 species, 739 segments) showed the teachers are
slightly over-confident — T=1.0 gives best calibration (log loss 0.1699 vs
0.3163 at T=0.5). Since calibration metrics don't capture downstream training
benefit, we use **T=1.0 (no scaling) for round 0** and defer T tuning.

### Implementation Details

All three data sources are combined into a single `ConcatDataset` and shuffled.
Each sample returns a `(waveform, labels, weight)` tuple. With batch_size=256
and natural proportions (~62% focal, <1% labeled soundscape, ~37% pseudo-labeled),
each batch contains a random mix of all sources.

**Per-sample weighted loss**: The `StudentFinetuneModule` computes per-sample BCE
with `reduction='none'`, then multiplies by the sample weight before averaging:

```python
per_sample_loss = BCE(logits, labels, reduction='none').mean(dim=1)  # (batch,)
loss = (per_sample_loss * weights).mean()                             # scalar
```

Ground-truth samples (weight=1.0) contribute full gradient signal. Pseudo-labeled
samples (weight=0.5) contribute half. This prevents noisy pseudo-labels from
dominating the loss.

**Soft targets and BCE**: BCE works natively with continuous targets in [0, 1].
For a pseudo-labeled sample where the teacher predicts P(species_X) = 0.72:

```
loss_X = -[0.72 × log(σ(logit_X)) + 0.28 × log(1 - σ(logit_X))]
```

The student learns to output ~0.72, preserving the teacher's uncertainty rather
than forcing a hard 0/1 decision. This is a form of knowledge distillation
where the student inherits the teacher ensemble's calibration.

**Validation and test sets contain only ground-truth data** (focal + labeled
soundscapes). No pseudo-labels in val/test. Early stopping on `val/auroc`
reflects real performance, not teacher-mimicking ability.

**EAT as student**: EAT was excluded from the exp 014 teacher ensemble due to
poor insect AUROC (0.673, caused by kaldi fbank fmax=8kHz). However, as a
student, EAT can learn from the ProtoCLR+ConvNeXt pseudo-labels which have
strong insect AUROC (0.904). The teacher ensemble's strength compensates for
EAT's spectral limitation during training.

### Variables

- **Independent**: Pseudo-label weight `w`, temperature T, confidence threshold,
  number of self-training iterations
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

- Exp 014 pseudo-labels (complete): 127K segments, ProtoCLR+ConvNeXt ensemble, AUROC 0.8851
- Exp 012 checkpoints (complete): ProtoCLR 0.767, ConvNeXt 0.755, EAT 0.720 LB
- Exp 012 training infrastructure: `train.py`, `MixedDataModule`, SLURM scripts (complete)

## Leads to

- Iterative self-training (round 1+): retrained students become teachers for exp 014
- SED architecture experiments: if student models plateau, try SED head (Phase A from roadmap)
- Ensemble inference: submit best students as an ensemble (Phase C from roadmap)
