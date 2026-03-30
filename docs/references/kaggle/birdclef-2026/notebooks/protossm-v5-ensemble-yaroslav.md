---
slug: "yaroslavkholmirzayev/birdclef-2026-protossm-v5-maximum-ensemble"
type: "notebook"
competition: "birdclef-2026"
date_captured: "2026-03-27"
votes: 0
---

# Kaggle Summary: BirdCLEF 2026 ProtoSSM v5 Maximum Ensemble

**Source**: https://www.kaggle.com/code/yaroslavkholmirzayev/birdclef-2026-protossm-v5-maximum-ensemble
**Competition**: BirdCLEF+ 2026
**Score**: 0.925 LB

## Summary

The highest-scoring public notebook in the ProtoSSM family lineage. Described as an
"operationalized faithful submit-path copy of the Pantanal Distill 0.924 notebook," this
submission pushes from 0.917 to 0.925 through three principal changes: (1) scaling ProtoSSM
to a wider d_model=256 with 3 SSM layers, (2) adding multi-head cross-attention after the
bidirectional SSM, and (3) a deep post-processing pipeline with 6 sequential stages
borrowed from 2025 Rank 1/3 solutions. The "v5" label and "maximum ensemble" title reflect
the accumulation of every incremental technique from v2 through v4 (cross-attention) plus
the full post-processing stack.

## Architecture

### What "v5" and "Maximum Ensemble" Mean

The notebook tracks an internal version numbering for ProtoSSM iterations:

| Version | d_model | Layers | Key addition | Notebook |
|---------|---------|--------|-------------|----------|
| v2 | 192 | 2 | Bidirectional SSM + prototypes + metadata | zirach (0.913) |
| v3 | 128 | 2 | Label smoothing + knowledge distillation | Pantanal Distill (0.917) |
| v4 | 256 | 3 | Cross-attention layer after SSM | (intermediate) |
| **v5** | **256** | **3** | **Full post-processing pipeline (6 stages)** | **This notebook (0.925)** |

Despite the class name remaining `ProtoSSMv2` in code, the actual configuration matches
what the author calls v4/v5 architecture. "Maximum ensemble" refers to combining every
available signal: ProtoSSM + MLP probes + ResidualSSM + genus proxies + metadata priors,
with OOF-optimized blend weights.

### Models Ensembled

The pipeline combines three model families at inference time:

1. **ProtoSSM v4** -- Bidirectional selective SSM with cross-attention, prototypical
   classification head, and gated fusion with Perch logits
2. **Per-class MLP probes** -- sklearn MLPClassifier on PCA-compressed Perch embeddings
   + context features (13 hand-engineered features per class)
3. **ResidualSSM** -- Lightweight 1-layer bidirectional SSM that predicts additive
   corrections to the first-pass ensemble

The final score is:

```
first_pass = W * ProtoSSM + (1 - W) * MLP_probes   # W optimized on OOF
final = first_pass + 0.3 * ResidualSSM(first_pass)
```

All three stages use the same Perch v2 frozen embeddings (1536-dim) as input.

## Key Techniques (0.917 to 0.925)

### 1. Wider ProtoSSM with Cross-Attention (v4 architecture)

The biggest single change from Pantanal Distill is scaling the SSM:

| Parameter | Pantanal Distill (v3) | This notebook (v4/v5) |
|-----------|----------------------|----------------------|
| d_model | 128 | **256** |
| n_ssm_layers | 2 | **3** |
| d_state | 16 | 16 |
| cross_attention | No | **Yes (4 heads)** |
| dropout | 0.15 | 0.15 |

The cross-attention layer (`TemporalCrossAttention`) is inserted after the bidirectional
SSM stack. It uses standard multi-head self-attention (4 heads) with a GELU FFN block,
allowing the model to capture non-local temporal patterns (dawn chorus onset,
counter-singing) that sequential SSM alone may miss.

### 2. Mixup + Focal Loss + SWA Training

Three training improvements layered on top of the v3 recipe:

- **File-level mixup** (alpha=0.3): Mixes pairs of files with Beta(0.3, 0.3) lambda,
  applied after epoch 5 warmup. Ensures lambda >= 0.5 so dominant sample stays dominant.
  Discrete metadata (site, hour) kept from the dominant sample.
- **Focal loss** (gamma=2.0): Reduces contribution of easy examples, focusing the model on
  hard-to-classify windows. Applied on top of pos_weight-balanced BCE.
- **SWA** (Stochastic Weight Averaging): Starts at 70% through training (swa_start_frac=0.7),
  accumulates checkpoint states, averages if >= 3 checkpoints. Falls back to best val loss
  checkpoint otherwise.

### 3. Six-Stage Post-Processing Pipeline

The full post-processing chain -- the "maximum" in the title:

```
ProtoSSM(pass1) + MLP -> ensemble -> ResidualSSM(pass2)
  -> TTA -> per-taxon temp -> file-level scale
  -> rank-aware -> delta smooth -> per-class sharpen
```

| Stage | Technique | Source | Config |
|-------|-----------|--------|--------|
| 1 | **TTA** (temporal shift) | New in v5 | shifts=[0, 1, -1], circular roll of 12-window sequence |
| 2 | **Per-taxon temperature** | v5 | Aves=1.10, Texture (Amphibia/Insecta)=0.95 |
| 3 | **File-level confidence scaling** | 2025 Rank 1 | top_k=2 windows, scale by top-K mean confidence |
| 4 | **Rank-aware scaling** | 2025 Rank 3 | power=0.5, scale by file_max^power |
| 5 | **Delta shift smoothing** | 2025 Rank 1 | alpha=0.15, temporal neighbor averaging |
| 6 | **Per-class threshold sharpening** | v5 | OOF-optimized F1 thresholds, sigmoid-style rescaling |

### 4. Genus Proxy Expansion

Automatic genus-level proxy mapping for unmapped species in Perch's label space. For any
competition species not in Perch's vocabulary, the system finds congeners (same genus) in
Perch's label set and uses max or mean of their logits as a proxy score. Applied to
Amphibia, Insecta, and Aves taxa. The proxy reduction method (max vs mean) is grid-searched
on OOF, defaulting to "max."

## Training and Inference Config

### ProtoSSM v4/v5 Training

| Parameter | Value |
|-----------|-------|
| n_epochs | 60 |
| lr | 1e-3 |
| weight_decay | 2e-3 |
| optimizer | AdamW |
| scheduler | OneCycleLR (pct_start=0.1, cosine) |
| grad_clip | norm=1.0 |
| patience | 15 (early stopping on val loss) |
| pos_weight_cap | 30.0 |
| distill_weight | 0.1 (MSE with Perch logits) |
| label_smoothing | 0.02 |
| mixup_alpha | 0.3 (after 5 warmup epochs) |
| focal_gamma | 2.0 |
| swa_start_frac | 0.7 |
| swa_lr | 5e-4 |
| OOF folds | 3-fold GroupKFold |

### MLP Probe Config

| Parameter | Value |
|-----------|-------|
| pca_dim | 64 |
| min_pos | 8 (minimum positive samples to train a probe) |
| hidden_layer_sizes | (128,) |
| activation | relu |
| max_iter | 300 |
| early_stopping | True (patience=15) |
| learning_rate_init | 0.001 |
| alpha (L2) | 0.01 |
| probe_blend_alpha | 0.40 (blend with base scores) |
| oversampling | Duplicate positives to balance classes |

### ResidualSSM Config

| Parameter | Value |
|-----------|-------|
| d_model | 64 |
| d_state | 8 |
| n_ssm_layers | 1 (bidirectional) |
| dropout | 0.1 |
| correction_weight | 0.3 |
| n_epochs | 30 |
| lr | 1e-3 |
| patience | 8 |
| target | residuals = labels - sigmoid(first_pass) |

### Prior Tables

Bayesian smoothing of site, hour, and site-hour label frequencies from labeled soundscapes.
Fold-safe (excludes validation files when computing priors). Fused into base scores with
separate lambda weights by taxon:

| Lambda | Value | Applied to |
|--------|-------|-----------|
| lambda_event | 0.4 | Mapped active Aves |
| lambda_texture | 1.0 | Mapped active Amphibia/Insecta |
| lambda_proxy_texture | 0.8 | Genus-proxy Amphibia/Insecta |
| smooth_texture | 0.35 | Temporal smoothing for texture taxa |
| smooth_event | 0.15 | Soft max-pool context for event birds |

### Inference Pipeline

1. Perch v2 inference: 12 x 5s windows per 60s soundscape (batch_files=16)
2. Prior fusion with site/hour/site-hour tables
3. ProtoSSM v4 forward pass with TTA (3 shifts)
4. MLP probe predictions blended into base scores (alpha=0.40)
5. Ensemble: OOF-optimized weight (default 0.5 ProtoSSM / 0.5 MLP)
6. ResidualSSM correction (weight=0.3)
7. Post-processing: 6 stages (temperature -> file-level -> rank-aware -> delta smooth -> sharpen)

## Post-Processing Pipeline (Detail)

### TTA (Test-Time Augmentation)

Circular-shifts the 12-window embedding sequence by [0, +1, -1] positions. Each shift
produces a ProtoSSM prediction, then the shifted predictions are rolled back to original
alignment before averaging. This is a lightweight TTA that requires no additional Perch
inference -- only 3 forward passes through the small ProtoSSM model.

### File-Level Confidence Scaling

For each file, compute the mean probability of the top-K windows (K=2). Multiply every
window's score by this mean. Effect: suppresses predictions in "quiet" files where even the
best windows are uncertain, boosts predictions in confident files.

### Rank-Aware Scaling (2025 Rank 3)

For each file, compute the per-class maximum across all 12 windows. Raise to power 0.5
(square root), then multiply each window's score. Effect: smooth suppression of uncertain
files without hard thresholding -- a softer version of file-level scaling.

### Delta Shift Smoothing (2025 Rank 1)

Temporal smoothing across consecutive windows within a file:
`new[t] = (1 - alpha) * old[t] + 0.5 * alpha * (old[t-1] + old[t+1])`
with alpha=0.15. Applied after rank-aware scaling. Different from the taxon-specific
smoothing applied earlier in the pipeline (which is part of prior fusion).

### Per-Class Threshold Sharpening

OOF-optimized per-class decision thresholds (grid over [0.20, 0.30, 0.40, 0.50, 0.60, 0.70])
maximizing per-class F1. Rather than hard thresholding, applies a sigmoid-style rescaling:
above-threshold scores are pushed toward 1.0, below-threshold toward 0.0. In submit mode,
defaults to threshold=0.5 for all classes (no OOF available).

## Key Ideas

- **Scale up the SSM**: Going from d_model=128 (Pantanal Distill) to d_model=256 with 3
  layers and cross-attention provides the single largest accuracy gain. The Perch embedding
  space is 1536-dim; a 128-dim bottleneck was too narrow.
- **Cross-attention after SSM**: Sequential SSMs excel at local temporal patterns but can
  miss non-local dependencies (e.g., dawn chorus onset patterns). Adding 4-head
  cross-attention after the SSM stack captures these global relationships.
- **Deep post-processing pipeline**: 6 sequential post-processing stages, each contributing
  small but additive gains. The combination of file-level scaling + rank-aware scaling +
  delta smoothing is borrowed from 2025 Rank 1 and Rank 3 solutions.
- **TTA without extra Perch cost**: Shifting the 12-window sequence is free since Perch
  embeddings are already computed. Only the lightweight ProtoSSM requires re-inference.
- **Two-pass boosting with ResidualSSM**: First pass creates a strong ensemble, second pass
  learns systematic correction patterns. ResidualSSM output head is zero-initialized so it
  starts as identity (no correction), learning only what helps.
- **OOF-honest everything**: Ensemble weights, per-class thresholds, probe hyperparameters,
  and proxy reduction method are all selected via OOF cross-validation with fold-safe prior
  tables that exclude validation files.
- **Per-class MLP probes with rich features**: Each class gets its own MLP trained on 13
  hand-engineered features (PCA embedding + sequential context + interaction terms + onset/
  offset detectors). Oversampling used to handle class imbalance.

## Results

| Metric | Value |
|--------|-------|
| Public LB | **0.925** |

### Progression in ProtoSSM Family

| Notebook | Score | Key Delta |
|----------|-------|-----------|
| perch-protossm-zirach | 0.913 | Baseline ProtoSSM v2 |
| Pantanal Distill | 0.917 | +ResidualSSM, +distillation, +label smoothing |
| **This (v5 Maximum Ensemble)** | **0.925** | +wider SSM, +cross-attn, +6-stage post-processing |

## Connections to Other Notebooks

- **Direct ancestor**: [pantanal-distill](pantanal-distill.md) (0.917) -- this notebook is
  described as an "operationalized faithful submit-path copy of the Pantanal Distill 0.924
  branch," meaning the code started as a fork of Pantanal Distill with v4/v5 upgrades
- **Same family**: [perch-protossm-zirach](perch-protossm-zirach.md) (0.913),
  [perch-protossm-fvegamaza](perch-protossm-fvegamaza.md) (0.913) -- the original ProtoSSM
  v2 implementations
- **Post-processing borrowed from**: 2025 Rank 1 (file-level confidence scaling, delta
  shift smoothing) and 2025 Rank 3 (rank-aware scaling) solutions
- **Perch backbone**: [perch-blamerx](perch-blamerx.md) (0.907) -- same Perch v2 feature
  extractor but without SSM temporal modeling
- **Probe approach**: [perch-probes-chiranjithdharma](perch-probes-chiranjithdharma.md) --
  per-class probe idea, extended here with richer features and MLP backend
