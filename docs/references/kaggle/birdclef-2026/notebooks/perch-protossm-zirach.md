---
slug: "zirach/birdclef-2026-perch-protossm"
type: "notebook"
competition: "birdclef-2026"
date_captured: "2026-03-24"
votes: 0
---

# Kaggle Summary: BirdCLEF 2026 Perch + ProtoSSM (zirach)

**Source**: https://www.kaggle.com/code/zirach/birdclef-2026-perch-protossm
**Competition**: BirdCLEF+ 2026
**Score**: 0.913 LB

## Summary

Foundation model pipeline using Google Perch v2 as a frozen feature extractor with
ProtoSSM (Prototypical State Space Model) for temporal modeling across 12 windows per
soundscape file. Combines ProtoSSM, per-class MLP probes, and site/hour metadata priors
in an ensemble. No training on focal recordings — only labeled soundscapes.

**Note**: This notebook is functionally identical to
[fvegamaza/birdclef-2026-perch-protossm](perch-protossm-fvegamaza.md) (only Unicode
formatting differences in comments).

## Approach

### Architecture

```
Soundscape (60s, 32kHz)
  → 12 × 5s windows
  → Perch v2 (frozen TF SavedModel)
    → per-window: embeddings (1536-dim) + logits (10K+ species)
  → Map to 234 BirdCLEF classes (203 direct + 31 genus proxies)
  → Three prediction streams:
      1. ProtoSSM: bidirectional SSM across 12 windows (temporal patterns)
      2. MLP probes: per-class classifiers on PCA embeddings + context features
      3. Metadata priors: site/hour species probability tables
  → Ensemble (50/50 ProtoSSM + MLP) → temperature scaling (T=1.10) → submission
```

### Perch v2 (Frozen Feature Extractor)

- Google's Bird Vocalization Classifier, TF SavedModel
- Outputs 1536-dim embeddings + logits for ~10K species per 5s window
- 203/234 BirdCLEF species map directly; 31 unmapped use genus-level proxies
  (max score across same-genus BirdClassifier species)
- Runs on CPU within 90-min Kaggle budget

### ProtoSSM (Novel Architecture)

Bidirectional selective state space model with prototypical classification:

| Parameter | Value |
|-----------|-------|
| d_input | 1536 (Perch embedding) |
| d_model | 128 |
| d_state | 16 |
| d_conv | 4 |
| n_ssm_layers | 2 (bidirectional) |
| dropout | 0.15 |
| Prototypes | 234 learnable (one per class) |
| proto_temp | ~5.0 (softplus) |

**Forward pass**:
1. Project embeddings: 1536 → 128 via Linear + LayerNorm + GELU
2. Add learnable positional encoding (12 positions)
3. 2× bidirectional SSM layers (forward + backward, merged, residual + LayerNorm)
4. Prototypical head: cosine similarity between hidden states and class prototypes
5. Gated fusion: `alpha * proto_sim + (1-alpha) * perch_logits` (per-class learned alpha)
6. Taxonomic auxiliary head: family-level classification for multi-task training

**Training**:
- 120 epochs, AdamW lr=2e-3, OneCycleLR, patience=20
- Multi-task loss: BCE (species) + 0.3 × MSE (Perch KD) + 0.1 × BCE (taxonomy)
- Trained on labeled soundscapes only (GroupKFold by file)

### Per-Class MLP Probes

- PCA compress Perch embeddings: 1536 → 64 dimensions
- Per-species sklearn MLPClassifier (128 hidden, ReLU, early stopping)
- Features: PCA embedding (64) + raw Perch score + prior score + base score +
  temporal context (prev/next window, file mean/max/std) + onset/offset diffs
- Prediction blended with base scores: `(1-0.4) * base + 0.4 * probe_logit`

### Metadata Priors

- Site-specific species probability tables from labeled soundscapes
- Hour-of-day priors (nocturnal vs diurnal patterns)
- Bayesian blending with exponential decay: `n/(n+8) * site_prior + (1-n/(n+8)) * global`
- Taxa-aware fusion weights: events (lambda=0.4), textures (lambda=1.0)

### Data Pipeline

- 60s soundscapes → 12 × 5s non-overlapping windows
- Perch inference batched at 16 files (192 windows) per batch
- No focal recordings used — trained exclusively on labeled soundscape segments

### Inference / Post-Processing

- **Temporal smoothing**: Different kernels for texture species (alpha=0.35) vs
  event species (alpha=0.15). Texture: neighbor averaging. Event: soft max-pool.
- **Temperature scaling**: T=1.10 before sigmoid (reduces overconfidence)
- **Ensemble**: 50% ProtoSSM + 50% MLP probes
- No TTA or model averaging

## Key Ideas

- **Perch v2 is extremely strong** — 0.825 LB out of the box, better than our trained SED
- **Temporal modeling across windows** captures dawn chorus patterns, species calling rates
- **Genus-level proxies** recover signal for 31 unmapped species using related Perch species
- **Taxa-aware processing** treats bird calls (transient events) differently from
  frog/insect sounds (continuous textures) — different smoothing and prior weights
- **No focal training needed** — the entire pipeline builds on frozen Perch features,
  with lightweight heads trained only on labeled soundscapes

## Results

| Metric | Value |
|--------|-------|
| Public LB | **0.913** |

## Connections

- **Related notebooks**: [perch-protossm-fvegamaza](perch-protossm-fvegamaza.md) (identical),
  [perch-blamerx](perch-blamerx.md) (same family), [pantanal-distill](pantanal-distill.md) (evolved version)
- **Existing analysis**: [discussions/perch-protossm-zirach.md](../discussions/perch-protossm-zirach.md)
- **Informs experiments**: exp 016 (SED architecture gap analysis)
