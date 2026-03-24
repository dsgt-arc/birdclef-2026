---
slug: "blamerx/birdclef-2026-inference"
type: "notebook"
competition: "birdclef-2026"
date_captured: "2026-03-24"
votes: 0
---

# Kaggle Summary: BirdCLEF 2026 Perch Inference (blamerx)

**Source**: https://www.kaggle.com/code/blamerx/birdclef-2026-inference
**Competition**: BirdCLEF+ 2026
**Score**: 0.907 LB

## Summary

Inference-only notebook for the Perch + ProtoSSM + ResidualSSM pipeline. This is the
**v8** version from blamerx, whose training notebook produces the artifacts consumed here.
Uses a two-pass architecture: first pass combines Perch + MLP probes + ProtoSSM ensemble,
second pass applies a ResidualSSM for error correction.

This is the **original author's** inference notebook — the zirach/fvegamaza notebooks
(0.913) appear to be a later version of the same system with different tuning.

## Approach

### Architecture

```
Pipeline: Perch → ProtoSSM(pass1) + MLP → ensemble → ResidualSSM(pass2) → final

Pass 1:
  Perch v2 (frozen) → 1536-dim embeddings + logits
  → Fuse with site/hour priors
  → MLP probes: per-class classifiers on PCA embeddings + context features
  → ProtoSSM: temporal modeling (if ENSEMBLE_WEIGHT > 0)
  → Blend: (1 - probe_alpha) * base + probe_alpha * probe_ensemble

Pass 2:
  ResidualSSM: (embeddings + pass1_scores) → learned corrections
  final = pass1 + correction_weight * correction
```

### Key Difference from zirach/fvegamaza

This notebook includes a **ResidualSSM second pass** — a lightweight bidirectional SSM
(d_model=64, d_state=8) that takes the first-pass scores + Perch embeddings and learns
per-species temporal error patterns. The correction is added with weight=0.3.

| Component | blamerx (v8) | zirach/fvegamaza |
|-----------|-------------|-----------------|
| ResidualSSM | Yes (d_model=64) | Not present |
| ProtoSSM config | From artifacts (config dict) | Inline code |
| Ensemble weight | From config (variable) | Fixed 0.5 |
| Probe alpha | From config (0.4) | 0.4 |
| Temperature | From config (1.10) | 1.10 |
| Score | 0.907 | 0.913 |

### ResidualSSM Architecture

```python
ResidualSSM(
    d_input=1536,        # Perch embedding dim
    d_scores=234,        # First-pass scores dim
    d_model=64,          # Lightweight hidden dim
    d_state=8,           # Smaller SSM state
    n_classes=234,
    n_windows=12,
    dropout=0.1,
    n_sites=20,          # Site embedding
)
```

- Input: concatenate embeddings (1536) + first-pass scores (234) = 1770-dim
- Project to 64-dim, add site/hour metadata embeddings
- Single bidirectional SSM layer
- Output head initialized to **zeros** (starts as identity, learns corrections)
- Applied as: `final = first_pass + 0.3 * residual_correction`

### Inference Pipeline

1. Load pre-computed artifacts from training notebook (`full_perch_arrays.npz`)
2. Run Perch v2 on test soundscapes (batch=16 files, 192 windows)
3. Fuse with site/hour priors (taxa-aware lambda weights)
4. Apply MLP + LR probe ensemble (per-class, PCA embedding features)
5. Run ResidualSSM second pass if available
6. Temperature scale (T=1.10), sigmoid, submit

### Data Pipeline

- Loads pre-serialized models from NPZ: MLP probes, LR probes, PCA, scaler, prior tables,
  ResidualSSM state dict — all pickled into byte arrays
- No training in this notebook — pure inference
- Artifacts from `blamerx/birdclef-2026-training` training notebook

### Post-Processing

Same as zirach/fvegamaza:
- Taxa-aware temporal smoothing: texture (alpha=0.35), events (alpha=0.15)
- Event smoothing uses soft max-pool (preserves transient peaks)
- Temperature scaling T=1.10

## Key Ideas

- **Two-pass boosting**: ResidualSSM learns systematic error patterns from the first pass
  ensemble — similar to gradient boosting in logit space
- **Artifact serialization**: All models (sklearn, PyTorch) serialized as byte arrays in
  a single NPZ file — elegant deployment pattern for Kaggle
- **Zero-initialized output head**: ResidualSSM starts as identity transform, gradually
  learns corrections — prevents catastrophic overriding of first-pass predictions

## Results

| Metric | Value |
|--------|-------|
| Public LB | **0.907** |

## Connections

- **Same family**: [perch-protossm-zirach](perch-protossm-zirach.md) (0.913),
  [pantanal-distill](pantanal-distill.md) (0.917)
- **Detailed analysis**: [perch-protossm-zirach-analysis.md](perch-protossm-zirach-analysis.md)
