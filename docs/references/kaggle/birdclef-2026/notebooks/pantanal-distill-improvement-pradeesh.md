---
slug: "pradeeshrajan/pantanal-distill-birdclef2026-improvement"
type: "notebook"
competition: "birdclef-2026"
date_captured: "2026-03-27"
votes: 0
---

# Kaggle Summary: Pantanal Distill BirdCLEF 2026 — Improvement (Pradeesh)

**Source**: https://www.kaggle.com/code/pradeeshrajan/pantanal-distill-birdclef2026-improvement
**Competition**: BirdCLEF+ 2026
**Score**: 0.926 LB (+0.009 over kamongi baseline at 0.917)

## Summary

A "V18" upgrade of the Pantanal Distill notebook (ashok205/kamongi, 0.917 LB). The
improvement comes from scaling up the ProtoSSM architecture, adding new training
techniques (CutMix, species-aware focal loss, SWA, cosine restarts), beefier MLP probes,
expanded TTA, and a multi-stage post-processing pipeline with isotonic calibration, rank-aware
scaling, and adaptive delta smoothing.

## What Changed: kamongi (0.917) vs Pradeesh (0.926)

### 1. ProtoSSM Architecture Scale-Up (V17 -> V18)

The single largest change: a much bigger ProtoSSM model with cross-attention.

| Parameter | kamongi (v2/v3) | Pradeesh (V18) |
|-----------|----------------|----------------|
| d_model | 128 | 320 |
| d_state | 16 | 32 |
| n_ssm_layers | 2 | 4 |
| n_prototypes | 1 | 2 |
| meta_dim | 16 | 24 |
| dropout | 0.15 | 0.12 |
| Cross-attention | None | 8-head TemporalCrossAttention |

The `TemporalCrossAttention` module is a new `nn.MultiheadAttention` block placed after the
bidirectional SSM layers. It captures non-local temporal patterns (e.g., dawn chorus onset,
counter-singing) that sequential SSM misses. Pre-norm residual architecture with a 2x FFN.

### 2. Training Recipe Overhaul

| Parameter | kamongi | Pradeesh (V18) |
|-----------|---------|----------------|
| n_epochs | 60 | 80 |
| lr | 1e-3 | 8e-4 |
| weight_decay | 2e-3 | 1e-3 |
| patience | 15 | 20 |
| pos_weight_cap | 30.0 | 25.0 |
| distill_weight | 0.1 | 0.15 |
| proto_margin | 0.1 | 0.15 |
| label_smoothing | 0.02 | 0.03 |
| oof_n_splits | 3 | 5 |
| mixup_alpha | 0.4 | 0.4 |
| focal_gamma | N/A | 2.5 |
| SWA | None | start at 65%, lr=4e-4 |
| LR schedule | OneCycleLR | CosineAnnealingWarmRestarts (T0=20) |

Key additions in training loop:
- **CutMix hybrid**: 30% probability of temporal CutMix (slice time windows), 70% standard
  Mixup. Both skip the first 5 warmup epochs.
- **Species-frequency-aware focal loss**: Per-class weights from inverse sqrt of frequency,
  combined with focal modulation (gamma=2.5). Replaces plain weighted BCE.
- **SWA (Stochastic Weight Averaging)**: Accumulates checkpoints from epoch 52+ (65% of 80),
  averages if >= 3 checkpoints collected.
- **Isotonic calibration + per-class threshold optimization**: New function
  `calibrate_and_optimize_thresholds()` that fits isotonic regression per class on OOF, then
  grid-searches F1-optimal thresholds.
- **Ensemble weight sweep**: Dedicated `sweep_ensemble_weight()` function.

### 3. Larger ResidualSSM (Second-Pass Boosting)

| Parameter | kamongi | Pradeesh |
|-----------|---------|----------|
| d_model | 64 | 128 |
| d_state | 8 | 16 |
| n_ssm_layers | 1 | 2 |
| correction_weight | 0.3 | 0.35 |
| n_epochs | 30 | 40 |
| lr | 1e-3 | 8e-4 |
| patience | 8 | 12 |

The kamongi version also grid-searches the correction_weight from OOF; the pradeesh version
uses a fixed 0.35.

### 4. MLP Probe Improvements

| Parameter | kamongi | Pradeesh |
|-----------|---------|----------|
| hidden_layer_sizes | (128,) | (256, 128) |
| max_iter | 300 | 500 |
| n_iter_no_change | 15 | 20 |
| learning_rate_init | 0.001 | 5e-4 |
| alpha (L2) | 0.01 | 0.005 |
| PCA dim | 64 | 128 |
| min_pos | 8 | 5 |
| Probe C | 0.50 | 0.75 |
| Probe alpha | 0.40 | 0.45 |

Pradeesh uses a 2-layer MLP (256 -> 128) instead of single-layer (128,), doubles the PCA
dimensionality, and lowers the minimum positive sample threshold from 8 to 5 (modeling more
rare species).

### 5. Post-Processing Pipeline (New in Pradeesh)

Kamongi uses a single uniform temperature (T=1.10) then writes submission.

Pradeesh adds a 5-step post-processing pipeline:
1. **Per-taxon temperature scaling**: Aves T=1.10, Texture T=0.95
2. **File-level confidence scaling**: Top-K=2 mean confidence multiplied per window
3. **Rank-aware scaling**: `scores * (file_max)^0.4` -- suppresses uncertain files
4. **Adaptive delta smoothing**: Confidence-dependent temporal smoothing (alpha=0.20 scaled
   by `1 - max_confidence`)
5. **Per-class threshold sharpening**: Push above-threshold scores higher, below-threshold
   lower (trained from OOF F1 optimization)

### 6. Test-Time Augmentation (TTA)

| Feature | kamongi | Pradeesh |
|---------|---------|----------|
| TTA shifts | None (single pass) | [0, 1, -1, 2, -2] |

Pradeesh circular-shifts the 12-window sequence by {+1, -1, +2, -2} steps, runs ProtoSSM
on each, then averages predictions (un-shifting before averaging).

### 7. Fusion Parameter Tuning

| Parameter | kamongi | Pradeesh |
|-----------|---------|----------|
| lambda_event | 0.4 | 0.45 |
| lambda_texture | 1.0 | 1.1 |
| lambda_proxy_texture | 0.8 | 0.9 |

### 8. What Pradeesh Dropped

- **YAMNet embeddings**: kamongi extracts YAMNet (1024-dim, 16kHz) and concatenates with
  Perch PCA for MLP probes. Pradeesh drops this entirely.
- **train_audio augmentation**: kamongi extracts Perch embeddings from individual species
  recordings (train_audio/) and uses them as extra positive examples for MLP probes.
  Pradeesh drops this.
- **Sonotype prior boost**: kamongi has `sonotype_prior_scale=2.0` that strengthens priors
  for insect sonotype classes. Pradeesh drops this.
- **Joint ensemble_weight x MLP alpha grid search**: kamongi does a 2D grid search over
  both ensemble weight and MLP alpha. Pradeesh does a simpler 1D sweep.
- **Correction weight grid search**: kamongi grid-searches the ResidualSSM correction weight
  from OOF. Pradeesh uses a fixed value.

## Architecture Diagram

```
Pass 1: Perch v2 -> ProtoSSM V18 (d=320, 4-layer BiSSM + 8-head cross-attn)
                  + MLP probes (256->128, PCA-128)
                  -> OOF-optimized ensemble blend
Pass 2: ResidualSSM (d=128, 2-layer BiSSM) correction (weight=0.35)
Post:   Per-taxon temp -> file confidence scale -> rank-aware scale
        -> adaptive delta smooth -> per-class threshold sharpen -> submission
```

## Key Ideas

- **Bigger model + more regularization**: The d_model jump (128 -> 320) with cross-attention
  provides much more capacity, counterbalanced by more epochs, lower dropout, SWA, CutMix,
  and focal loss to prevent overfitting.
- **Cross-attention captures non-local patterns**: The sequential SSM can only propagate
  information step-by-step through the 12-window sequence. Cross-attention allows direct
  interaction between any pair of windows.
- **Multi-stage post-processing**: Each step is individually small but they compound.
  Rank-aware scaling is particularly interesting -- it suppresses predictions in files where
  the model is generally uncertain.
- **Species-frequency focal loss**: Rare species get both higher class weights (inverse sqrt
  frequency) and focal modulation (easy examples down-weighted). This replaces the simpler
  pos_weight-only approach.
- **SWA for free lunch**: Averaging checkpoints from the last 35% of training smooths the
  loss landscape at no extra inference cost.

## What Likely Matters Most for the +0.009

Based on the magnitude of changes:

1. **ProtoSSM scale-up + cross-attention** (~0.004-0.005): The 2.5x wider model with 4 SSM
   layers and 8-head cross-attention is the single biggest architectural change.
2. **Post-processing pipeline** (~0.002-0.003): Rank-aware scaling + adaptive delta
   smoothing + per-class thresholds are three new post-processing steps that compound.
3. **Training improvements** (~0.001-0.002): Focal loss, CutMix, SWA, 5-fold OOF, and
   cosine warm restarts all contribute small gains.
4. **MLP probe upgrade** (~0.001): Deeper MLP, higher PCA dim, lower min_pos.
5. **TTA** (~0.001): 5-shift TTA vs no TTA.

## Results

| Metric | Value |
|--------|-------|
| Public LB | **0.926** |

## Connections

- **Parent**: [pantanal-distill](pantanal-distill.md) (0.917) -- this is a direct fork
- **Same family**: [perch-protossm-zirach](perch-protossm-zirach.md) (0.913),
  [perch-blamerx](perch-blamerx.md) (0.907)
- **Key lesson**: Within the Perch + ProtoSSM paradigm, scaling up the SSM model and adding
  cross-attention yields the biggest gains. The post-processing pipeline provides meaningful
  additional improvement.
