---
slug: "chiranjithdharma/0-910-score"
type: "notebook"
competition: "birdclef-2026"
date_captured: "2026-03-24"
votes: 0
---

# Kaggle Summary: 0.910 Score (chiranjithdharma)

**Source**: https://www.kaggle.com/code/chiranjithdharma/0-910-score
**Competition**: BirdCLEF+ 2026
**Score**: 0.910 LB

## Summary

Frozen Perch v2 with per-class logistic regression probes and strong Bayesian
site/hour priors. No neural temporal modeling (no ProtoSSM/SSM) — achieves 0.910
with just sklearn LogisticRegression on PCA-compressed Perch embeddings. The
simplest high-scoring approach in this competition.

## Approach

### Architecture

```
Soundscape (60s, 32kHz)
  → 12 × 5s windows
  → Perch v2 (frozen TF SavedModel)
    → per-window: embeddings (1536-dim) + logits (14,795 classes)
  → Map to 234 BirdCLEF classes (direct + genus proxies for amphibians)
  → Bayesian fusion with site/hour priors
  → Per-class logistic regression probes on PCA embeddings + context features
  → Dynamic alpha blending → temporal smoothing → sigmoid → submission
```

### Perch v2 (Frozen)

- Google's Bird Vocalization Classifier, TF SavedModel
- 14,795 output classes, 1536-dim embeddings
- Species mapping: direct match on scientific name, genus-level proxy for
  unmapped amphibians (max logit across same-genus Perch species)
- CPU-only inference

### Bayesian Prior Fusion

Separate handling by taxa type — the key differentiator of this notebook:

| Parameter | Value | Applied to |
|-----------|-------|------------|
| lambda_event | 0.4 | Bird classes (Perch already informative) |
| lambda_texture | 1.0 | Amphibia/Insecta (location-determined) |
| lambda_proxy_texture | 0.8 | Unmapped genus proxies |
| smooth_texture | 0.35 | Gaussian temporal smoothing on textures |

Prior tables built from labeled soundscapes:
- Global species prevalence
- Per-site prevalence (P(species | site))
- Per-hour prevalence (P(species | hour_utc))
- Site × hour interaction
- Bayesian smoothing: `n/(n+k) * local_prior + (1-n/(n+k)) * global` (k=8 for site/hour, k=4 for interaction)

### Per-Class Logistic Regression Probes

For each species with ≥8 positive training samples:

- **Input features** (~72 dim): PCA-compressed Perch embedding (32-dim) +
  raw Perch logit + prior logit + fused base logit + sequential context
  (prev window, next window, file mean, file max)
- **Model**: sklearn LogisticRegression (C=0.25, liblinear, balanced weights)
- **OOF validation**: 5-fold GroupKFold by site (entire sites held out)
- **Dynamic alpha blending**: `alpha = 0.25 + 0.25 * min(1.0, (n_pos - 8) / 40)`
  — more training data → more trust in the probe

### Data Pipeline

- 59 fully-labeled training soundscapes (12 windows each = 708 samples)
- No focal recordings used
- Optional Perch cache from `jaejohn/perch-meta` dataset

### Post-Processing

- **Temporal smoothing**: Gaussian kernel [0.1, 0.2, 0.4, 0.2, 0.1] on texture
  classes within each 60s file (α=0.35)
- **No temperature scaling** (unlike ProtoSSM notebooks that use T=1.10)
- **No TTA or model averaging**

## Key Ideas

- **Simplicity wins**: 0.910 LB with just LogisticRegression on frozen embeddings
  — no PyTorch, no GPU training, no SSM/attention
- **Dynamic probe blending**: Per-class alpha based on training data quantity
  prevents overfitting probes for rare species while leveraging them for common ones
- **Taxa-aware fusion weights**: Birds get light prior influence (λ=0.4) since
  Perch already handles them well; frogs/insects get strong priors (λ=1.0) since
  they're more location-dependent
- **Genus proxies for amphibians**: Unmapped frogs use max logit across same-genus
  Perch species — recovers signal without any training
- **Site-level OOF folds**: GroupKFold by site prevents data leakage in prior tables
  (critical since site is the strongest predictor for texture species)

## Results

| Metric | Value |
|--------|-------|
| Public LB | **0.910** |

## Connections

- **Same family**: [perch-protossm-zirach](perch-protossm-zirach.md) (0.913),
  [pantanal-distill](pantanal-distill.md) (0.917)
- **Key difference**: No ProtoSSM/SSM temporal modeling — uses LogReg probes instead.
  Simpler but nearly matches ProtoSSM (0.910 vs 0.913)
- **Shared infrastructure**: Same Perch v2 model, same cache dataset (jaejohn/perch-meta),
  same TF 2.20 wheels
