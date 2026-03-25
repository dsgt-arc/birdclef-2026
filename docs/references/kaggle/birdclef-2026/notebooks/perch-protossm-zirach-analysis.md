---
title: "BirdCLEF 2026 Perch + ProtoSSM Analysis (0.913 LB)"
source: "https://www.kaggle.com/code/zirach/birdclef-2026-perch-protossm"
date_captured: "2026-03-24"
score: 0.913
---

# Perch + ProtoSSM Notebook Analysis (0.913 LB)

Public notebook achieving 0.913 on BirdCLEF 2026 leaderboard.

## Architecture Overview

This is NOT a trained-from-scratch SED model like our exp 016. It's a
**foundation model pipeline** that uses Google Perch v2 as a frozen feature
extractor, then builds multiple prediction layers on top:

```
Soundscape audio (1 min, 32kHz)
  → Split into 12 × 5s windows
  → Perch v2 (frozen TF model) → per-window embeddings (1536-dim) + logits
  → Three prediction streams fused:
      1. ProtoSSM: State space model across 12 windows (temporal modeling)
      2. MLP probes: Per-class classifiers on PCA-compressed embeddings
      3. Metadata priors: Site/hour/month probability tables
  → Ensemble fusion → temperature scaling → submission
```

## Key Components

### 1. Perch v2 Foundation Model (Frozen)

- Google's Bird Vocalization Classifier, TensorFlow SavedModel
- Outputs per-window: embeddings (1536-dim) + logits for ~10K species
- Maps 203/234 BirdCLEF 2026 species directly
- **Unmapped species** (31): genus-level proxies from related Perch species
- Runs on CPU within 90-min Kaggle budget

### 2. ProtoSSM (Novel — Prototypical State Space Model)

A lightweight PyTorch model that processes the **sequence of 12 windows per file**:

- **Input**: Perch embeddings (12 × 1536) + Perch logits (12 × 234) per file
- **Selective SSM** (Mamba-style): Models temporal patterns across the 12 windows
  (dawn chorus onset, species-specific calling rates, territorial singing patterns)
- **Prototypical classification head**: Learnable species prototypes, cosine similarity
- **Multi-task training**:
  - Species BCE loss (primary)
  - Prototypical contrastive loss (margin=0.1)
  - Knowledge distillation from Perch logits (weight=0.3)
  - Taxonomic auxiliary loss (family-level classification)
- **Config**: d_model=128, d_state=16, 2 SSM layers, dropout=0.15
- **Training**: 120 epochs, lr=2e-3, patience=20, trained on labeled soundscapes only

### 3. Per-Class MLP/LogReg Probes

- PCA compress Perch embeddings: 1536 → 64 dimensions
- Per-species classifiers with rich features:
  - PCA-compressed embedding (64-dim)
  - Raw Perch score, prior score, base score
  - Temporal context: prev/next window, file mean/max/std
  - Onset/offset detection: diff from previous/next window
  - Interaction terms
- 5-fold GroupKFold (by file) for honest OOF evaluation
- MLP with (128,) hidden layer, or LogisticRegression as fallback

### 4. Metadata Priors

- Site-specific species probability tables
- Hour-of-day priors (nocturnal vs diurnal)
- Site × hour interaction tables
- Built from labeled soundscape annotations, fold-safe

### 5. Score Fusion

Three prediction streams combined:
```python
# Weighted fusion of ProtoSSM + MLP probes + metadata priors
lambda_event = 0.4        # Aves species (transient calls)
lambda_texture = 1.0      # Amphibia/Insecta (continuous textures)
lambda_proxy_texture = 0.8
smooth_texture = 0.35     # Temporal smoothing for texture species
smooth_event = 0.15       # Lighter smoothing for event species
```

**Taxa-aware processing**: Different smoothing for birds (events, lighter
smoothing to preserve transient calls) vs amphibians/insects (textures,
heavier smoothing for continuous vocalizations).

### 6. Post-Processing

- **Taxa-specific temporal smoothing**: Neighbor averaging with different alpha
  for events (0.15) vs textures (0.35)
- **Event smoothing**: Soft max-pool (preserves transient peaks unlike averaging)
- **Temperature scaling**: Applied before submission

## What Makes This Score 0.913

### It's NOT training a classifier from scratch

This notebook doesn't train on focal recordings at all. It uses Perch v2 as
a frozen 1536-dim feature extractor and builds lightweight prediction heads
on top. The 0.913 comes from:

1. **Perch v2 is extremely strong** (0.825 LB out-of-the-box)
2. **ProtoSSM adds temporal modeling** across the 12 windows per file
3. **Per-class probes** refine predictions using embeddings + context features
4. **Metadata priors** (site, hour) constrain predictions to ecologically plausible species
5. **Taxa-aware fusion** treats birds differently from frogs/insects
6. **Smart unmapped species handling** via genus-level Perch proxies

### Key Differences from Our Approach

| Aspect | Us (Exp 016) | Perch + ProtoSSM |
|--------|-------------|-----------------|
| **Foundation model** | None (train from scratch) | Perch v2 (frozen, 10K+ species) |
| **Backbone** | EfficientNet-B0 (3.6M params, ImageNet) | Perch v2 (~100M params, bird audio) |
| **Training data** | Focal + labeled soundscapes | Labeled soundscapes only (for probes) |
| **Temporal modeling** | None (per-window independent) | SSM across 12 windows per file |
| **Metadata features** | None | Site, hour, month priors |
| **Species mapping** | Direct 234-class head | 203 direct + 31 genus proxies |
| **Inference framework** | ONNX (PyTorch) | TensorFlow (Perch) + PyTorch (ProtoSSM) |
| **LB score** | 0.795 | 0.913 |

## Implications for Our Work

### What we can learn

1. **Perch v2 as feature extractor**: The 1536-dim Perch embeddings are incredibly
   powerful. Even a simple linear probe on Perch embeddings beats our trained SED model.

2. **Temporal modeling across windows**: Our SED model processes each 5s/10s window
   independently. This notebook models the sequence of 12 windows per file, capturing
   patterns like dawn chorus progression. SSM/Mamba architecture is lightweight and
   CPU-friendly.

3. **Metadata priors are free performance**: Site-specific species pools and hour-of-day
   patterns constrain predictions to ecologically plausible species. We have this
   metadata in the filenames but don't use it.

4. **Taxa-aware processing**: Different temporal smoothing for transient bird calls vs
   continuous frog/insect textures. Simple but effective.

5. **Genus-level proxies**: For unmapped species, use related species in the Perch
   taxonomy as proxies. Recovers signal for species without direct classification support.

### What we should consider

1. **Hybrid approach**: Use Perch v2 embeddings as features for our SED model, or
   ensemble our SED predictions with Perch-based predictions.

2. **Add temporal context**: Our current per-window SED inference misses file-level
   temporal patterns. Even simple neighbor smoothing (which we added) helps, but
   a learned temporal model (SSM/Mamba) would be stronger.

3. **Add metadata priors**: Parse site/hour from soundscape filenames, build species
   probability tables from labeled data, multiply into predictions.

4. **Consider Perch as baseline**: If Perch alone = 0.825 and our best = 0.795,
   we might get further faster by building on top of Perch than training from scratch.
