---
slug: "marynaborovska/birdclef2026-optimized"
type: "notebook"
competition: "birdclef-2026"
date_captured: "2026-03-27"
votes: 0
---

# Kaggle Summary: BirdCLEF 2026 Optimized (marynaborovska)

**Source**: https://www.kaggle.com/code/marynaborovska/birdclef2026-optimized
**Competition**: BirdCLEF+ 2026
**Score**: 0.921 LB

## Summary

The highest-scoring public Perch + ProtoSSM notebook at time of capture. Builds on the
pantanal-distill (0.917) foundation with seven additive techniques: focal loss, mixup
augmentation, cross-attention after SSM, stochastic weight averaging (SWA), per-taxon
temperature scaling, file-level confidence scaling, and temporal-shift TTA. Internally
labeled as "V17 research-optimized hyperparameters over V16 (0.917)."

## Approach

### Architecture

```
Soundscape (60s, 32kHz)
  -> 12 x 5s windows
  -> Perch v2 (frozen TF SavedModel)
    -> per-window: embeddings (1536-dim) + logits (14,795 classes)
  -> Map to 234 BirdCLEF classes (direct + genus proxies for Amphibia/Insecta/Aves)
  -> Three prediction streams:
      1. ProtoSSM v4: BiSSM + cross-attention + prototypical head (temporal)
      2. MLP probes: per-class MLPClassifier on PCA embeddings + context features
      3. Metadata priors: site/hour/site-hour Bayesian tables
  -> Ensemble (OOF-optimized weight, default 50/50)
  -> ResidualSSM second-pass correction (weight=0.3)
  -> TTA (temporal shifts [0, +1, -1])
  -> Per-taxon temperature scaling (Aves=1.10, Texture=1.05)
  -> File-level confidence scaling (top-K=2)
  -> Submission
```

### ProtoSSM v4 Architecture

Extends v3 (pantanal-distill) with a cross-attention layer. Full config:

| Parameter | Value | Notes |
|-----------|-------|-------|
| d_model | 128 | Hidden dimension |
| d_state | 16 | SSM state size |
| n_ssm_layers | 2 | Bidirectional SSM layers |
| dropout | 0.15 | |
| n_prototypes | 1 | Per-class prototype |
| n_sites | 20 | Site embedding vocab |
| meta_dim | 16 | Metadata embedding dim |
| use_cross_attn | True | **NEW vs v3** |
| cross_attn_heads | 4 | Multi-head attention |

Architecture flow:
1. Linear projection (1536 -> 128) + LayerNorm + GELU + dropout
2. Learnable positional encoding (12 positions)
3. Metadata: site embedding + hour embedding -> linear projection, added to hidden state
4. 2-layer bidirectional SelectiveSSM (Mamba-style input-dependent discretization)
5. **NEW: TemporalCrossAttention** -- multi-head self-attention (4 heads) with pre-norm
   residual + FFN (GELU, 2x expansion). Captures non-local patterns (dawn chorus onset,
   counter-singing) that sequential SSM may miss
6. Prototypical cosine similarity (per-class prototypes, learned temperature) + class bias
7. Gated fusion with Perch logits (per-class learned sigmoid gate)
8. Optional taxonomic auxiliary head (family-level)

### MLP Probes

Per-class MLPClassifier (sklearn) for species with >=8 positive samples:

| Parameter | Value |
|-----------|-------|
| hidden_layer_sizes | (128,) |
| activation | relu |
| max_iter | 300 |
| early_stopping | True |
| validation_fraction | 0.15 |
| n_iter_no_change | 20 |
| learning_rate_init | 0.001 |
| alpha (L2) | 0.001 |

Input features per class (~77 dim): PCA-compressed Perch embedding (64-dim default) +
raw logit + prior logit + fused base logit + prev/next/mean/max/std window context +
diff_mean/diff_prev/diff_next (onset/offset detection) + 3 interaction terms
(raw*prior, raw*base, prior*base). Oversampling of positives for class balance.

### Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| n_epochs | 60 | |
| lr | 1e-3 | With OneCycleLR (pct_start=0.1, cosine) |
| weight_decay | 2e-3 | |
| val_ratio | 0.15 | |
| patience | 12 | Early stopping |
| pos_weight_cap | 30.0 | Clamped class weights |
| distill_weight | 0.1 | MSE loss vs Perch logits |
| label_smoothing | 0.05 | Increased from 0.02 (pantanal) |
| oof_n_splits | 3 | GroupKFold |
| **mixup_alpha** | **1.0** | File-level embedding mixup, skips first 5 warmup epochs |
| **focal_gamma** | **2.0** | Focal loss replacing weighted BCE |
| **swa_start_frac** | **0.7** | Start SWA at 70% of training |
| **swa_lr** | **5e-4** | SWA learning rate |

Loss function: `focal_BCE(gamma=2.0, pos_weight) + 0.1 * MSE_distill + 0.1 * family_BCE`

### ResidualSSM (Second-Pass Boosting)

Lightweight correction model trained on first-pass residuals:

| Parameter | Value |
|-----------|-------|
| d_model | 64 |
| d_state | 8 |
| n_ssm_layers | 1 |
| dropout | 0.1 |
| correction_weight | 0.3 |
| n_epochs | 30 |
| lr | 1e-3 |
| patience | 8 |

Takes concat(embeddings, first_pass_scores) as input, predicts additive corrections.
Trained with MSE loss on residuals (target - sigmoid(first_pass)). Output initialized
near zero. Skipped if wall time exceeds 4 minutes (time-safety for Kaggle budget).

### Pre-Computed Cache

Uses `jaejohn/perch-meta` dataset for pre-computed Perch embeddings and logits:
- `full_perch_meta.parquet` -- row-level metadata (row_id, filename, site, hour)
- `full_perch_arrays.npz` -- scores_full_raw + emb_full arrays
- `full_oof_meta_features.npz` -- cached OOF base/prior features
- Falls back to live Perch inference if cache not found (train mode only)
- Submit mode requires cache (raises FileNotFoundError otherwise)

## Key Techniques: +0.004 over Pantanal-Distill (0.917 -> 0.921)

The seven additions that collectively push from 0.917 to 0.921:

### 1. Focal Loss (gamma=2.0)

Replaces weighted BCE with focal loss. Down-weights easy negatives (the dominant class
for rare species), focusing gradient on hard examples. Combined with per-class pos_weight
(capped at 30x). This is the single most impactful change for rare species performance.

```python
focal_weight = (1 - pt) ** gamma  # pt = p for positives, 1-p for negatives
loss = focal_weight * bce
```

### 2. File-Level Mixup (alpha=1.0)

Mixes pairs of files at the embedding level using Beta(1.0, 1.0) = Uniform[0,1]
distribution (clamped so lambda >= 0.5). Applied after 5 warmup epochs. Mixes embeddings,
logits, and labels; discrete metadata (site, hour) kept from dominant sample. Alpha=1.0
is aggressive -- the uniform distribution maximizes regularization.

### 3. Cross-Attention After SSM

A `TemporalCrossAttention` module inserted after the bidirectional SSM layers:
- Multi-head self-attention (4 heads, batch_first)
- Pre-norm residual connection
- FFN with GELU activation, 2x expansion, dropout
- Captures non-local temporal patterns the sequential SSM scan may miss

### 4. Stochastic Weight Averaging (SWA)

Accumulates model checkpoints starting at 70% of training (epoch 42/60). If >=3
checkpoints accumulated, loads their arithmetic mean as the final model. Falls back to
best-validation-loss checkpoint otherwise. Improves generalization by averaging across
the loss landscape.

### 5. Per-Taxon Temperature Scaling

Different temperature per taxa class at inference:
- **Aves**: T=1.10 (wider/softer -- birds have more uncertainty)
- **Insecta/Amphibia** (texture): T=1.05 (sharper -- these are more location-determined)

Applied after ensemble but before sigmoid: `probs = sigmoid(logits / T_per_class)`

### 6. File-Level Confidence Scaling

Described as "Rank 1/2 technique." For each file (12 windows), computes the top-K (K=2)
mean prediction per class across windows, then multiplies each window's probability by
this file-level confidence. Effect: suppresses noise in uncertain files, boosts
predictions where the model is consistently confident.

```python
sorted_view = np.sort(view, axis=1)  # sort along windows
top_k_mean = sorted_view[:, -top_k:, :].mean(axis=1, keepdims=True)
scaled = view * top_k_mean
```

### 7. Temporal-Shift TTA

Test-time augmentation by circular-shifting the 12-window embedding sequence by
[0, +1, -1] positions. Each shifted sequence gets a full ProtoSSM forward pass, then
predictions are unshifted and averaged. Provides robustness to temporal alignment.

## Post-Processing Pipeline (in order)

1. **Ensemble fusion**: `w * ProtoSSM + (1-w) * MLP_probes` (w from OOF grid search,
   default 0.5)
2. **ResidualSSM correction**: `final += 0.3 * correction` (if trained within time budget)
3. **TTA**: Average predictions from 3 temporal shifts [0, +1, -1]
4. **Per-taxon temperature**: Divide logits by T (Aves=1.10, texture=1.05) before sigmoid
5. **File-level confidence scaling**: Multiply probs by top-2 mean confidence per file
6. **Clip**: Clamp to [0, 1]
7. **Temporal smoothing**: Applied within fuse_scores_with_tables before ensemble --
   Gaussian neighbor smoothing (alpha=0.35 for texture, soft max-pool alpha=0.05 for events)

## Key Ideas

- **Focal loss is critical for rare species**: With 234 classes and only ~59 labeled files,
  most classes have very few positives. Standard BCE is dominated by easy negatives. Focal
  loss (gamma=2.0) combined with pos_weight gives rare species a fighting chance.
- **Aggressive mixup (alpha=1.0) regularizes small datasets**: With only ~59 training files,
  overfitting is the primary failure mode. Uniform-distributed mixup at the embedding level
  provides strong regularization without needing additional data.
- **Cross-attention complements SSM**: The sequential SSM scan captures local temporal
  dependencies (adjacent windows), but non-local patterns (species appearing in windows 1
  and 11 but not between) need attention. Adding cross-attention after SSM captures both.
- **SWA over best-checkpoint**: Instead of selecting a single best checkpoint, averaging
  multiple late-training checkpoints produces a more robust model (especially with small
  data where validation loss is noisy).
- **File-level confidence scaling is a strong prior**: If a species has high confidence in
  2+ windows of a file, it probably really is present -- boosting all windows suppresses
  false negatives. If confidence is low everywhere, multiplying by a small top-K mean
  further suppresses false positives.
- **Per-taxon temperature acknowledges different calibration needs**: Aves predictions from
  Perch+ProtoSSM tend to be overconfident (T=1.10 softens); texture classes are more
  reliable and only need slight compression (T=1.05).

## Results

| Metric | Value |
|--------|-------|
| Public LB | **0.921** |

## Connections

- **Direct ancestor**: [pantanal-distill](pantanal-distill.md) (0.917) -- same pipeline
  with ProtoSSM v3 + ResidualSSM + MLP probes + prior fusion. This notebook adds the
  seven techniques above.
- **Grandparent**: [perch-protossm-zirach](perch-protossm-zirach.md) (0.913) -- original
  ProtoSSM v2 + MLP probe ensemble. No ResidualSSM, no distillation, no focal loss.
- **Baseline**: [chiranjithdharma](perch-probes-chiranjithdharma.md) (0.910) -- LogReg
  probes only, no ProtoSSM/SSM. The +0.011 gap from 0.910 to 0.921 comes from: ProtoSSM
  temporal modeling (+0.003 via zirach), ResidualSSM + distillation tuning (+0.004 via
  pantanal), and the seven focal/mixup/cross-attn/SWA/temp/file-scale/TTA additions
  (+0.004 via this notebook).
- **Shared infrastructure**: Same Perch v2 model, same cache dataset (jaejohn/perch-meta),
  same TF 2.20 wheels (ashok205/tf-wheels)
- **Progressive improvement chain**:
  chiranjithdharma (0.910) -> zirach (0.913) -> pantanal-distill (0.917) -> **this (0.921)**
