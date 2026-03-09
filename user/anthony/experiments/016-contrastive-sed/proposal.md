# Proposal: Prototypical SED with Frame-Level Similarity

**Status**: proposed
**Created**: 2026-03-08
**Author**: Anthony

## Hypothesis

**Primary**: Framewise max aggregation of per-frame predictions on frozen
EfficientNet-B1 features improves clip-level cmAP over mean-pooled prediction
(the linear probe baseline at 0.336 val/cmap, 0.649 Kaggle private).

**Secondary**: A prototypical classification head (cosine similarity against species
prototypes) can match a linear probe while enabling prototype-swappable inference.

## Background

### Motivation

Experiments 014 and 015 showed that teacher-generated soft labels degrade performance
compared to hard labels when training a frozen EfficientNet-B1:

| Experiment | Approach | val/cmap |
|-----------|----------|----------|
| 012 | Linear probe, hard labels | **0.336** |
| 014 | Linear probe, soft labels | 0.238 |
| 015 | SED head, soft labels | 0.237 |

The bottleneck is label quality, not architecture. Meanwhile, top BirdCLEF 2025
solutions all use SED (+0.03-0.05 gain), but they train with hard labels on
trainable backbones.

### Key insight

The question is really two separable questions:

1. **Does framewise max help?** Apply per-frame predictions and take the max over
   frames instead of mean-pooling. This catches brief calls that get diluted by
   mean pooling. All top BirdCLEF 2025 solutions use this.
2. **Can prototypical classification match a linear probe?** Replace learned weight
   vectors with mean-of-embeddings prototypes. Less expressive, but enables
   prototype swapping (few-shot, domain adaptation).

These are tested independently via ablation baselines (see Procedure).

### Relationship to ProtoCLR and prototypical networks

This experiment is a **frame-level extension of prototypical networks** (Snell et al.,
2017) applied to SED. Key differences from standard approaches:

| | Prototypical Networks | ProtoCLR | This experiment |
|---|---|---|---|
| **Granularity** | Clip-level embeddings | Clip-level embeddings | **Frame-level** embeddings |
| **Episodes** | Small N-way K-shot | Small N-way K-shot | **Full 206-class** every step |
| **Prototypes** | Computed per episode | Computed per episode | Computed from all training data, **re-computed every N epochs** |
| **Loss** | Softmax over distances | InfoNCE contrastive | **BCE** with hard labels (multi-label compatible) |
| **Backbone** | Learned end-to-end | Learned end-to-end | **Frozen** + projection head |
| **SED output** | No | No | **Per-frame similarity scores** → framewise max |

The novelty is the frame-level extension — per-frame cosine similarity with framewise
max aggregation for SED — not the metric learning itself.

### Related work

- **Exp 011** (Prompted SED): Tested prompted SED with SSAMBA backbone. Phase 0 found
  the backbone representations were too weak (24% probe accuracy). The prototypical
  similarity head was identified as more promising than concatenation-based prompting.
- **Prototypical networks** (Snell et al., 2017): Few-shot classification via
  distance to class prototypes. Our approach extends this to per-frame SED.
- **ProtoCLR**: Combines prototypical networks with contrastive learning. Uses
  episodic training with InfoNCE loss. We use full 206-class BCE instead,
  avoiding the sparse supervision problem of episodic training.
- **BirdCLEF 2025 winners**: All used SED + attention pooling for training, framewise
  max for inference. We adopt this train/infer divergence pattern.

## Method

### Approach

Process each clip through a frozen EfficientNet-B1 backbone to get frame-level
features. A lightweight projection head maps frames into an embedding space.
Compute cosine similarity between each frame and all 206 species prototypes.
BCE loss on the clip-level prediction (framewise max of similarities, then sigmoid).
Prototypes are re-computed from training data every epoch.

### Architecture

```
Training clip (5s)
      |
      v
  [Mel Spec + SpecAugment]
  (B, 1, 256, 625)
      |
      v
+--------------------------+
| EfficientNet-B1 (frozen) |
+--------------------------+
      |
      v
  (B, 1280, 8, 20)
      |
  mean(dim=2)  <- freq pool
      |
      v
  (B, 1280, 20)
      |
  transpose -> (B, 20, 1280)
      |
      v
+--------------------------+
| Projection Head          |
| Linear(1280, D)          |
| -> LayerNorm -> ReLU     |
| -> Linear(D, D)          |
| -> L2-normalize          |   clamp norm >= 1e-8 before dividing
+--------------------------+
      |
      v
  query_frames (B, 20, D)
      |
      +--- cosine sim against prototypes (206, D) ---+
      |                                               |
      v                                               |
  frame_scores (B, 20, 206)   <- similarity / tau     |
      |                                               |
      +----------------+----------------+             |
      |                |                |             |
  [Train: framewise   [Val: framewise  [Inference:    |
   max -> sigmoid      max -> cmAP     framewise max  |
   -> BCE loss]        evaluation]     -> sigmoid     |
                                       -> submit]     |

Prototypes (206, D):
  Re-computed every N epochs from training data.
  For each species: mean of L2-normalized frame embeddings
  across all training chunks, then L2-normalize the mean.
```

~660K trainable params (projection head). Backbone frozen.

### Key design decisions

**Prototypical classification, not pairwise contrastive**: An earlier version of this
proposal used pairwise episode sampling (50% positive, 50% negative pairs). Review
identified that this gives 186x sparser supervision than the linear probe (1 species
per episode vs 206 classes per sample). The prototypical classification formulation
gives the same supervision density as the linear probe while preserving the metric
learning properties.

**L2-normalization with numerical safety**: All embeddings are L2-normalized before
similarity computation. Norm is clamped to >= 1e-8 to prevent gradient explosion on
near-zero-norm vectors (e.g., silence frames from the frozen backbone).

**Fixed temperature**: Temperature `tau` is fixed at 1.0 (not 0.07). The original
CLIP-style temperature of 0.07 is designed for InfoNCE loss with large negative sets.
For cross-entropy over cosine similarities in [-1, 1], tau=0.07 saturates the softmax
and kills gradients. Start with tau=1.0 and tune as a hyperparameter if needed.

**BCE loss, not cross-entropy**: Focal recordings are single-species, but soundscape
inference is multi-label (multiple species per window). Cross-entropy/softmax trains
mutually exclusive predictions — a fundamental train/test mismatch. BCE treats each
species independently, which is consistent with the multi-label inference requirement.
This also matches exp 012's loss (focal BCE with gamma=2).

**No reference/query split during training**: Unlike the pairwise formulation, there
is no "reference clip" during training. Each training sample is classified against
pre-computed prototypes for all 206 species. References only appear at inference time
when computing prototypes for few-shot species.

**Framewise max, not mean pool**: Frame-level similarity → max over frames → clip
prediction. This catches species present in only a few frames (brief calls) better
than mean pooling, which dilutes the signal. This matches the dominant pattern from
top BirdCLEF 2025 solutions.

**Deferred for v0**: Reference attention pooling, clip attention pooling, learnable
temperature, few-shot holdout evaluation. These are enhancements to add if v0 shows
promise.

### Setup

| Component | Details |
|-----------|---------|
| Data | 28,564 focal recordings (BirdCLEF 2025), 206 species |
| Backbone | EfficientNet-B1 (DBD-research-group/EfficientNet-B1-BirdSet-XCL), frozen |
| Trainable params | Projection head only (~660K) |
| Compute | 1x RTX 6000 (24GB), ~2-4 hours |
| Dependencies | exp 012 codebase (data loading, mel pipeline) |
| Validation | Same 10% held-out split as exp 012/014 (seed=42) |

### Procedure

1. **Cosine gap analysis** (30 min, before anything else):
   - Run exp 011 Phase 0 analysis on the EfficientNet-B1 backbone.
   - Extract frame embeddings for ~200 clips (20 species × 10 clips).
   - Compute within-species vs between-species cosine similarity gap.
   - Report silhouette score and linear probe accuracy.
   - This tells us if the backbone has *any* metric structure worth building on.
   - For comparison: SSAMBA had gap of 0.04 (within=0.445, between=0.405).

2. **Zero-training baseline** (1 hour, before writing training code):
   - Run all training and val clips through the frozen backbone.
   - Freq-pool → (1280, 20) → L2-normalize each frame.
   - For each species: mean-pool all training frame embeddings → prototype (1280,).
   - For each val clip: cosine similarity to all 206 prototypes.
   - Report val/cmap **three ways** (these are the key ablation baselines):
     - **A. Nearest centroid (clip-level)**: mean-pool frames, cosine sim → cmAP
     - **B. Nearest centroid (frame-level)**: per-frame cosine sim, max → cmAP
     - **C. Linear probe with framewise max**: `Linear(1280, 206)` applied per-frame,
       max over frames, trained with BCE. Same as exp 012 but with framewise max.
   - **Decision gate**:
     - If A and B both < 0.10: no metric structure. Abort.
     - If B > A by >0.03: framewise max adds value. Proceed.
     - If B ≈ A: no SED benefit. Framewise max is not the answer.
     - If C > exp 012 (0.336): framewise max is the winning factor, not prototypes.
       Just use a linear probe with framewise max (simpler, more expressive).

3. **Data preparation**:
   - Reuse exp 012's focal data pipeline. Each recording yields multiple 5s chunks.
   - Build a species-indexed lookup for prototype computation.
   - Cache backbone features for all chunks (avoid redundant forward passes).

4. **Prototype computation**:
   - For each species: gather all training chunk frame embeddings, mean-pool, L2-norm.
   - Store as `prototypes (206, D)` tensor.
   - Re-compute **every epoch** (not every 5 epochs — with cached backbone features,
     this only requires a projection head forward pass over cached features, ~1 second).

5. **Training**:
   - Standard data loader (not episodic) — same as exp 012.
   - Forward pass: backbone (frozen, no_grad) → projection head → L2-norm → frame embeddings.
   - Per-frame cosine similarity: `frame_embeds @ prototypes.T / tau` → `(B, 20, 206)`.
   - Clip logits: `max(dim=1)` → `(B, 206)`.
   - Loss: **BCE (focal, gamma=2)** with hard species labels (matches exp 012).
   - LR: 1e-3, cosine schedule, 30 epochs.
   - Batch size: 256 (same as exp 012).
   - Fixed tau=1.0.

6. **Early diagnostics** (after epoch 2):
   - Compute mean intra-species cosine sim vs inter-species in projected space.
   - If gap < 0.05: projection head isn't learning. Check gradients, try higher LR.
   - Compare val/cmap at epoch 2 vs zero-training baseline — should be improving.

7. **Evaluation**:
   - val/cmap using framewise max.
   - Compare against full ablation table:
     - Exp 012: linear probe, mean pool (0.336)
     - Baseline C: linear probe, framewise max (from step 2)
     - Baseline A: nearest centroid, clip-level (from step 2)
     - Baseline B: nearest centroid, frame-level (from step 2)
     - **This experiment**: prototypical + projection, framewise max
   - Per-species AP comparison to identify winners/losers.
   - Qualitative frame-level visualization: per-frame similarity heatmaps overlaid
     on spectrograms for 5-10 examples. Use BirdNET/Perch predictions (exp 010)
     as rough temporal reference for qualitative assessment.

8. **ONNX export + Kaggle submission**:
   - Export backbone + projection head as ONNX.
   - Pre-compute and cache 206 species prototypes as numpy array.
   - Per soundscape window: project frames, matrix multiply with cached prototypes,
     framewise max → clip predictions.

### Variables

- **Independent**: Training formulation (prototypical vs linear probe),
  projection dimension D (256, 512, 1280), temperature tau
- **Dependent**: val/cmap, Kaggle private score, per-species AP
- **Controlled**: Backbone (frozen EfficientNet-B1), data split (seed=42),
  mel spectrogram params (n_mels=256, hop=256)

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| val/cmap | Class-mean average precision on all 206 species |
| Kaggle private | Private leaderboard score |
| per-species AP | AP breakdown to identify species where prototypical helps/hurts |
| cosine gap | Mean intra-species cosine sim minus inter-species (embedding quality) |
| zero-training cmap | Baseline cmAP with no trained projection (decision gate) |

### Baseline

| Baseline | val/cmap | Source |
|----------|----------|--------|
| A. Nearest centroid, clip-level (no training) | TBD | Step 2 |
| B. Nearest centroid, frame-level (no training) | TBD | Step 2 |
| C. Linear probe + framewise max (trained) | TBD | Step 2 |
| Exp 012: linear probe + mean pool (trained) | 0.336 | Exp 012 |
| Exp 014: linear probe + soft labels | 0.238 | Exp 014 |
| Exp 015: SED + soft labels | 0.237 | Exp 015 |

### Success Criteria

- **Abort if**: cosine gap < 0.02 on EfficientNet-B1 (step 1) — no metric structure
- **Abort if**: baselines A and B both < 0.10 (step 2) — no prototype viability
- **Reframe if**: B ≈ A — no SED benefit from framewise max, wrong hypothesis
- **Reframe if**: baseline C > 0.336 — framewise max helps but prototypes aren't needed, just use linear probe with framewise max
- **Confirm if**: prototypical + projection > baseline C — prototype formulation adds value beyond framewise max alone
- **Reject if**: prototypical + projection < 0.25 — prototypes are too weak

## Risks and Mitigations

### Supervision signal

The prototypical formulation gives 206-class supervision per sample (same as linear
probe), eliminating the 186x sparsity problem of the earlier pairwise design. However,
the prototype-derived logits are less expressive than learned weight vectors — the
prototypes are averaged embeddings, never directly optimized for classification.

### Projection bottleneck

The projection head reduces dimensionality from 1280 to D. If D is too small, it
discards discriminative information the linear probe uses freely. Mitigation: test
D=1280 (no bottleneck, just a learned rotation) as an ablation.

### Prototype staleness

Prototypes are re-computed every epoch. Between updates within an epoch, the projection
head changes but prototypes lag. This is an alternating optimization — the projection
head optimizes against fixed prototypes, then prototypes are re-derived from the
updated projection. With cached backbone features, re-computing prototypes takes ~1
second (just the projection head forward pass over cached features). Every-epoch
refresh minimizes staleness. If training is unstable at epoch boundaries, reduce LR
or try EMA prototype updates as a follow-up.

### Frozen backbone temporal structure

The key bet is that EfficientNet-B1's (1280, 8, 20) feature map has useful temporal
structure at the 250ms frame level. If it doesn't, framewise max collapses to mean
pool and there's no SED benefit. The zero-training baseline (step 1) directly tests
this before any training investment.

### Comparison fairness

The linear probe learns 206 dedicated weight vectors optimized end-to-end for
classification. The prototypical approach learns a shared embedding space and derives
"classifiers" from data averages. Even if prototypical cmAP is lower, the few-shot
capability and temporal localization may be independently valuable.

## Limitations

- **Frozen backbone**: Cannot adapt features to the metric learning objective. LoRA
  or partial unfreezing is a natural follow-up if the projection head is insufficient.
- **No temporal ground truth**: Frame-level SED quality is assessed via (a) framewise
  max vs mean pool cmAP gap as a temporal discriminativeness proxy, (b) qualitative
  spectrogram overlays, and (c) rough comparison against BirdNET/Perch temporal
  predictions from exp 010. These are proxies, not ground truth evaluation.
- **Prototype quality**: Mean-pooled prototypes collapse intra-class variation (call
  types, recording conditions). K=5 reference clips at inference may not capture species
  diversity. Sensitivity analysis (different prototype sets) should quantify this.
- **Prototypes are less expressive than learned weights**: The linear probe's weight
  vectors are directly optimized for classification. Prototypes are data averages,
  never directly optimized. This is a fundamental expressiveness gap that the
  projection head must compensate for.
