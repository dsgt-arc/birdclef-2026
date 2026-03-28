---
slug: "yuriygreben/birdclef-2026-protossm-v5-time-optimised"
type: "notebook"
competition: "birdclef-2026"
date_captured: "2026-03-28"
votes: 0
---

# Kaggle Summary: BirdCLEF 2026 ProtoSSM v5 Time Optimised

**Source**: https://www.kaggle.com/code/yuriygreben/birdclef-2026-protossm-v5-time-optimised
**Competition**: BirdCLEF+ 2026
**Score**: 0.925 LB
**Author**: yuriygreben

## Summary

A self-contained, time-optimised ProtoSSM v5 notebook that trains and infers entirely
within a single Kaggle kernel. Unlike the Yaroslav "maximum ensemble" variant (also
0.925), this notebook includes the complete training code inline, making the full
pipeline transparent and reproducible. It uses a pre-computed Perch cache
(`jaejohn/perch-meta`) for the labeled soundscapes but runs live Perch inference on
test soundscapes. Key innovation: all model training (ProtoSSM, MLP probes,
ResidualSSM, prior tables) happens inside the notebook on the 66 labeled soundscapes,
with OOF cross-validation for hyperparameter selection.

## Architecture Overview

Three-model ensemble with two-pass error correction:

```
Perch v2 (frozen) → embeddings (1536-dim) + logits (14795-dim)
                         ↓                        ↓
                    ProtoSSM v5 ←── gated fusion ──┘
                         ↓
                  MLP probes (per-class)
                         ↓
       ensemble: W * ProtoSSM + (1-W) * MLP (OOF-optimized W)
                         ↓
              ResidualSSM (second-pass boosting)
                         ↓
              6-stage post-processing pipeline
                         ↓
                   submission.csv
```

## ProtoSSM v5 (v4 in code)

d_model=256, d_state=16, 3 BiSSM layers + 4-head cross-attention.

**Components**:
1. **Input projection**: Linear(1536 → 256) + LayerNorm + GELU + Dropout(0.15)
2. **Positional encoding**: Learnable (1, 12, 256)
3. **Metadata embeddings**: Site Embedding(20, 16) + Hour Embedding(24, 16) → Linear(32 → 256)
4. **3x Bidirectional SSM layers**: Each has forward SSM + backward SSM (flipped) → Linear(512 → 256) + LayerNorm + residual
5. **Cross-attention** (after SSM): 4-head MultiheadAttention + FFN(256→512→256, GELU) with pre-norm residual
6. **Prototypical classification**: Cosine similarity to learnable prototypes (234, 256) × learned temperature + per-class bias
7. **Gated fusion**: `alpha * ProtoSSM_logits + (1-alpha) * Perch_logits` where alpha is per-class learned sigmoid gate
8. **Taxonomic auxiliary head**: Linear(256 → n_families) on mean-pooled temporal features

**SelectiveSSM** (Mamba-style): in_proj(2D), conv1d(d_model, groups=d_model, kernel=4), SiLU, dt_proj, A_log, B_proj, C_proj → sequential scan → out_proj + skip

## Training

**Multi-task loss**:
- Primary: Focal BCE (gamma=2.0) with class pos_weight (capped at 30)
- Distillation: 0.1 * MSE(output, Perch_logits)
- Taxonomic: 0.1 * BCE(family_head, family_labels)

**Training config**:
- 60 epochs, lr=1e-3, AdamW (weight_decay=2e-3)
- OneCycleLR (pct_start=0.1, cos anneal)
- Label smoothing: 0.02
- File-level MixUp (alpha=0.3, skip first 5 warmup epochs, lambda≥0.5)
- SWA: start at 70% of epochs, average if ≥3 checkpoints
- Early stopping: patience=15, monitor val_loss
- Gradient clip: 1.0
- 3-fold OOF GroupKFold by site

**Prototype initialization**: Class-mean embeddings from training data (normalized).

## MLP Probes

Per-class sklearn MLPClassifier on PCA-compressed embeddings.

**Features per class** (d=PCA_dim + 13):
- PCA-compressed embeddings (64 dims, StandardScaler → PCA)
- Raw Perch score (1)
- Prior score (1)
- Base fused score (1)
- Temporal context: prev_base, next_base, mean_base, max_base, std_base (5)
- Difference features: diff_mean, diff_prev (onset), diff_next (offset) (3)
- Interaction terms: raw*prior, raw*base, prior*base (3)

Total: 77 features (64 PCA + 13 context/interaction)

**MLP config**: hidden=(128,), ReLU, max_iter=300, early_stopping=True, validation_fraction=0.15, lr_init=0.001, alpha=0.01

**Class balancing**: Positive oversampling (repeat positives to match negatives).

**Probe blending**: `(1 - 0.30) * base_score + 0.30 * probe_logit` (alpha=0.30)

**Frozen best probe params**: pca_dim=64, min_pos=8, C=0.15, alpha=0.30

## ResidualSSM (Second-Pass Boosting)

Lightweight SSM that corrects first-pass ensemble errors.

**Architecture**:
- Input: concat(embeddings 1536 + first_pass_scores 234) = 1770-dim
- Projection: Linear(1770 → 64) + LayerNorm + GELU + Dropout(0.1)
- Metadata: Site Embedding(20, 8) + Hour Embedding(24, 8) → Linear(16 → 64)
- Positional encoding: Learnable (1, 12, 64)
- 1x Bidirectional SSM layer (d_model=64, d_state=8)
- Output: Linear(64 → 234), **zero-initialized** (corrections start small)

**Training**: 30 epochs, lr=1e-3, MSE loss on residuals (target - sigmoid(first_pass)), patience=8, 15% validation split.

**Application**: `final = first_pass + 0.3 * correction`

**Wall-time safety**: Skips if >4 min elapsed (preserves time for test inference).

## Species Mapping

- Direct mapping via taxonomy.csv → Perch labels.csv join on scientific_name
- Automatic genus proxies for unmapped non-sonotype species (Amphibia, Insecta, Aves)
- Genus proxy reduction: max (default, grid-searchable between max/mean)
- Unmapped inactive species: score = -8.0 (suppressed)
- Prior-only species: Use metadata priors without Perch signal

## Metadata Priors

Bayesian smoothing of site/hour/site-hour frequency tables from labeled data.

**Smoothing**: `w = n / (n + k)`, where k_site=8, k_hour=8, k_interaction=4.
Prior applied as **additive in logit space**: `scores += lambda * prior_logits`

**Lambda weights**: event (Aves) = 0.4, texture (Amphibia/Insecta) = 1.0, proxy = 0.8

**Fold-safe**: Prior tables re-fit for each OOF fold, excluding validation files.

## Score Fusion Pipeline

1. ProtoSSM v5 inference with TTA (shifts [0, +1, -1])
2. Prior-fused base scores via metadata tables
3. MLP probe scores (PCA features + context + interactions)
4. Ensemble: `W * ProtoSSM + (1-W) * MLP` (W=0.5 default, OOF-optimized in train mode)
5. ResidualSSM correction: `+ 0.3 * residual_ssm_output`

## Post-Processing Pipeline (6 stages)

1. **Per-taxon temperature**: Aves T=1.10, Texture T=0.95 → sigmoid
2. **File-level confidence scaling**: Multiply each window by top-K=2 mean confidence per file
3. **Rank-aware scaling** (2025 Rank 3): `probs *= file_max^0.5`
4. **Delta shift smoothing** (2025 Rank 1): `(1-0.15)*x + 0.075*(prev + next)`
5. **Per-class threshold sharpening**: OOF-optimized thresholds remap scores
   - Above threshold: `0.5 + 0.5 * (score - t) / (1 - t)`
   - Below threshold: `0.5 * score / t`

## Cache Architecture

**Labeled soundscapes**: Loaded from `jaejohn/perch-meta` cache or computed live:
- `full_perch_meta.parquet`: row_id, filename, site, hour_utc
- `full_perch_arrays.npz`: scores_full_raw (N, 234), emb_full (N, 1536)

**Test soundscapes**: Always live Perch inference (batch_files=16).

## Key Differences from Other ProtoSSM Notebooks

| Feature | This notebook | Yaroslav v5 (0.925) | Pantanal-distill (0.917) |
|---------|--------------|---------------------|--------------------------|
| Training code | **Inline (visible)** | Pre-trained artifacts only | Pre-trained artifacts only |
| ProtoSSM d_model | 256 | 256 | 128 |
| SSM layers | 3 | 3 | 2 |
| Cross-attention | 4-head | 4-head | None |
| Probe backend | **MLP** (sklearn) | MLP | MLP / LogReg |
| Probe features | **77-dim** (PCA64 + 13 context/interaction) | PCA64 + context | PCA64 + context |
| ResidualSSM | Yes (d=64) | Yes (d=64) | Yes (d=64) |
| SWA | Yes (start 70%) | Yes | No |
| Focal loss | Yes (gamma=2.0) | Yes | No |
| File-level MixUp | Yes (alpha=0.3) | Yes | No |
| TTA shifts | [0, +1, -1] | [0, +1, -1] | None |
| Post-processing | 6 stages | 6 stages | 3 stages |
| Threshold sharpening | OOF-optimized per-class | OOF-optimized | None |
| Wall-time safety | ResidualSSM skip if >4min | Unknown | None |

## What Makes This "Time Optimised"

1. **Batch Perch inference** (batch_files=16) instead of file-by-file
2. **Wall-time safety**: ResidualSSM skipped if >4 min elapsed
3. **Submit mode fast path**: Skips OOF CV, probe grid, diagnostics
4. **Frozen hyperparameters**: Best probe params pre-set, no grid search in submit
5. **Pre-computed cache** for labeled soundscapes (only test runs live)

## Data Sources

- Competition: `birdclef-2026`
- Perch model: `google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1`
- TF 2.20 wheels: attached dataset
- Perch cache: `jaejohn/perch-meta` (pre-computed labeled soundscape embeddings)

## Runtime

- Perch on ~600 test files: ~80 min (batch_files=16)
- ProtoSSM training (60 epochs): ~10s
- MLP probes: ~5s
- ResidualSSM training (30 epochs): ~3s
- Total: fits within 90 min per kernel (single kernel, no chain needed with cache)

## Key Takeaways

1. **All training in-notebook**: No cross-environment sklearn issues — PCA, probes, ProtoSSM, ResidualSSM all trained and applied in the same Kaggle session
2. **77-dim probe features**: Richer than our 37-dim (adds std, diff features, interactions)
3. **Ensemble weight optimization**: OOF grid search for ProtoSSM vs MLP blend weight
4. **Focal loss + MixUp + SWA**: Three training enhancements stacked on ProtoSSM
5. **6-stage post-processing**: Each stage adds incremental value, borrowed from 2025 winners
6. **Single-kernel architecture**: Uses pre-computed cache for labeled data, runs Perch on test live — no two-notebook chain
