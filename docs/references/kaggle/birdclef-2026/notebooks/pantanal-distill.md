---
slug: "ashok205/pantanal-distill-birdclef2026"
type: "notebook"
competition: "birdclef-2026"
date_captured: "2026-03-24"
votes: 0
---

# Kaggle Summary: Pantanal Distill BirdCLEF 2026

**Source**: https://www.kaggle.com/code/ashok205/pantanal-distill-birdclef2026
**Competition**: BirdCLEF+ 2026
**Score**: 0.917 LB (highest public score among these notebooks)

## Summary

The most evolved version of the Perch + ProtoSSM pipeline family. Builds on the same
core architecture (Perch v2 frozen feature extractor + ProtoSSM temporal modeling + MLP
probes + metadata priors) but adds ProtoSSM v3 improvements, explicit knowledge
distillation from Perch, and ResidualSSM second-pass boosting. The "distill" in the name
refers to knowledge distillation from Perch logits into the ProtoSSM student model.

## Approach

### Architecture

```
Pass 1: Perch v2 → ProtoSSM v3 + MLP probes → ensemble (60/40)
Pass 2: ResidualSSM correction (weight=0.3)
Final: temperature scaling (T=1.10) → sigmoid → submission
```

### What's New vs zirach/fvegamaza (0.913)

| Feature | zirach (0.913) | Pantanal Distill (0.917) |
|---------|---------------|------------------------|
| ProtoSSM version | v2 | v3 |
| ResidualSSM | Not present | Yes (second-pass boosting) |
| Knowledge distillation | weight=0.3 | weight=0.1 (tuned down) |
| Ensemble weights | 50/50 | 60/40 (ProtoSSM/MLP) |
| Training epochs | 120 | 60 (faster convergence) |
| OOF folds | 5-fold | 3-fold (speed tradeoff) |
| Early stopping patience | 20 | 15 |
| Label smoothing | None | 0.02 |

### ProtoSSM v3 Architecture

Same base structure as v2 (bidirectional SSM + prototypical head + gated fusion) with:

| Parameter | Value |
|-----------|-------|
| d_model | 128 |
| d_state | 16 |
| n_ssm_layers | 2 |
| dropout | 0.15 |
| meta_dim | 16 (site + hour embeddings) |
| n_sites | 20 |

**Key additions**:
- Site embedding: `Embedding(20, 16)` — recording location encoded into SSM
- Hour embedding: `Embedding(24, 16)` — time-of-day encoded into SSM
- Metadata projection: `Linear(32, 128)` added to SSM hidden state
- Label smoothing: 0.02 in BCE loss

### Knowledge Distillation

The "distill" technique: ProtoSSM is trained to match Perch logits (teacher) while also
fitting ground truth labels (student):

```python
loss_bce = BCE_with_logits(species_out, labels, pos_weight=pos_weight)
loss_distill = MSE(species_out, perch_logits)  # Perch as soft teacher
loss_family = BCE_with_logits(family_out, family_labels)  # taxonomic aux

loss = loss_bce + 0.1 * loss_distill + 0.1 * loss_family
```

This keeps ProtoSSM aligned with Perch's strong species representations while learning
complementary temporal patterns from the SSM layers.

### ResidualSSM (Second-Pass Boosting)

Lightweight SSM that learns to correct systematic errors from the first-pass ensemble:

```python
ResidualSSM(
    d_input=1536,     # Perch embeddings
    d_scores=234,     # First-pass ensemble scores
    d_model=64,       # Half the size of ProtoSSM
    d_state=8,
    n_classes=234,
    n_windows=12,
    dropout=0.1,
    n_sites=20,
)
# Output head initialized to zeros (starts as identity)
# Applied: final = first_pass + 0.3 * correction
```

Training: 30 epochs, lr=1e-3, patience=8. Trained on residuals (errors) from first pass.

### Data Pipeline

- Same as other ProtoSSM variants: Perch v2 inference on 12 × 5s windows
- Trained on labeled soundscapes only (GroupKFold, 3 splits)
- MLP probes with PCA-compressed embeddings (64-dim) + context features

### Training

**ProtoSSM v3**:
- 60 epochs, AdamW lr=1e-3, weight_decay=2e-3
- OneCycleLR (pct_start=0.1, cosine annealing)
- Gradient clipping: norm=1.0
- Early stopping: patience=15 on val loss
- Class weighting: pos_weight capped at 30.0

**MLP Probes**: Same as zirach (sklearn MLPClassifier, 128 hidden, early stopping)

### Inference / Post-Processing

- **Ensemble**: 60% ProtoSSM + 40% MLP probes (optimized on OOF)
- **ResidualSSM**: correction_weight=0.3
- **Temporal smoothing**: texture (alpha=0.35), events (alpha=0.15)
- **Temperature scaling**: T=1.10
- **Total runtime**: ~6 min on Kaggle (wall time)

## Key Ideas

- **Two-stage boosting**: First pass creates an ensemble, second pass (ResidualSSM) learns
  species-specific and temporal error patterns — like gradient boosting in logit space
- **Metadata in SSM**: Site and hour embeddings directly injected into SSM hidden states
  (not just used as prior tables) — the model learns site-specific temporal patterns
- **Gated fusion alpha**: Per-class learned weight between ProtoSSM prototypical similarity
  and raw Perch logits — automatically learns when to trust Perch vs learned features
- **OOF-optimized weights**: Ensemble blend ratio (60/40) chosen via OOF validation rather
  than manual tuning
- **Conservative distillation**: weight=0.1 (lower than zirach's 0.3) — enough to keep
  alignment but doesn't dominate the BCE objective

## Results

| Metric | Value |
|--------|-------|
| Public LB | **0.917** |

## Connections

- **Same family**: [perch-protossm-zirach](perch-protossm-zirach.md) (0.913),
  [perch-blamerx](perch-blamerx.md) (0.907)
- **Key difference**: ResidualSSM second pass + metadata in SSM + tuned ensemble weights
  account for the +0.004 over zirach
