---
name: "Mix2: Mixture of Mixups"
source: "https://arxiv.org/abs/2403.09598"
license: "CC BY-NC-ND 4.0"
category: "data-augmentation"
pipeline_stage:
  - "training"
relevance: "Multi-label augmentation for rare species in bioacoustics — directly applicable to BirdCLEF class imbalance"
status: "reference"
---

# Mix2: Mixture of Mixups for Multi-label Classification of Rare Anuran Sounds

**Paper**: Moummad, Farrugia, Serizel, Froidevaux, Lostanlen (2024)
**arXiv**: [2403.09598](https://arxiv.org/abs/2403.09598)
**Code**: [github.com/ilyassmoummad/Mix2](https://github.com/ilyassmoummad/Mix2)
**Venue**: arXiv preprint (cs.SD), March 2024

## Overview

Mix2 addresses multi-label imbalanced classification in bioacoustics by combining
three mixing regularization methods — Mixup, Manifold Mixup, and MultiMix — into
a single framework that randomly selects one method per training iteration. The key
finding: individual mixing methods often underperform baseline, but their random
combination yields significant gains, especially for rare classes (+6.2% F-score on
rare species vs baseline). Applied to anuran (frog/toad) sound classification using
the AnuraSet dataset.

## Relevance to BirdCLEF 2026

Our competition shares the same challenges:
- **Multi-label**: avg 4.2 species per 5s soundscape segment
- **Class imbalance**: 82/206 focal species have <100 recordings; 25 insect sonotypes
  are soundscape-only with 6-168 segments each
- **Species co-occurrence**: dense nocturnal frog choruses, daytime bird+insect overlap
- **BCE loss**: same loss function as Mix2

We currently use spectrogram-space MixUp only. Mix2 adds:
1. **Manifold Mixup** — mixing in embedding space (post-backbone), which may produce
   more semantically meaningful augmentations than raw spectrogram mixing
2. **MultiMix** — batch-level Dirichlet interpolation for broader augmentation diversity
3. **Random alternation** — avoiding overfitting to any single mixing strategy

## Method

### Three mixing strategies

**1. Mixup (input-space)**

Standard interpolation of input spectrograms and labels:
```
x_mix = lambda * x_i + (1 - lambda) * x_j
y_mix = lambda * y_i + (1 - lambda) * y_j
```
Where `lambda ~ Beta(alpha, alpha)`.

**2. Manifold Mixup (embedding-space)**

Same interpolation but applied to intermediate hidden representations:
```
h_mix = lambda * h_i + (1 - lambda) * h_j
y_mix = lambda * y_i + (1 - lambda) * y_j
```
Where `h_i, h_j` are embeddings at a randomly selected intermediate layer.
This smooths decision boundaries in learned representation space rather than
pixel space.

**3. MultiMix (batch-level)**

Extends mixing to the entire mini-batch using matrix operations:
```
H_mix = Lambda @ H
Y_mix = Lambda @ Y
```
Where `Lambda` is sampled from a Dirichlet distribution (not Beta). This creates
convex combinations across all batch members simultaneously, sampling from the
"entire convex hull of the mini-batch" rather than just pairwise segments.

### Mix2: random selection per iteration

At each SGD training step, randomly select one of the three methods:
- **Mixup**: 25% probability
- **Manifold Mixup**: 50% probability
- **MultiMix**: 25% probability

Label mixing is performed at the loss level (soft targets with BCE).

The 50% weight on Manifold Mixup was empirically determined (paper does not
justify the specific distribution).

## Architecture and training

| Component | Specification |
|-----------|---------------|
| Model | MobileNetV3-Large (3M params, trained from scratch) |
| Input | 16 kHz, 3s segments, 128 mel filters, 512-sample window, 128 hop |
| Loss | Binary Cross-Entropy (BCE) |
| Optimizer | AdamW (lr=0.01, wd=1e-6) |
| Batch size | 128 |
| Epochs | 100 |
| Other augmentation | SpecAugment + Random Circular Time Shift |
| Metric | Macro F-score (equal weight per species) |

## Results on AnuraSet

### Full AnuraSet (93,378 segments, 42 species)

| Method | Frequent | Common | Rare | All |
|--------|----------|--------|------|-----|
| ResNet18 (11M, ImageNet pretrained) | 61.6 | 52.0 | 14.8 | 34.9 |
| ResNet152 (116M, ImageNet pretrained) | 68.4 | 56.8 | 15.7 | 37.8 |
| MBN baseline (3M, scratch, 10 ep) | 86.3 | 72.4 | 35.4 | 48.2 |
| MBN + 100 epochs | 88.9 | 75.5 | 45.6 | 56.2 |
| MBN + SpecAug + TimeShift | 87.0 | 74.2 | 46.9 | 56.6 |
| + Mixup only | 87.4 | 74.3 | 44.9 | 55.2 |
| + Manifold Mixup only | 86.8 | 74.3 | 47.0 | 56.6 |
| + MultiMix only | 88.0 | 75.1 | 46.1 | 56.3 |
| + Mixup + Manifold Mixup | 88.6 | 76.6 | 51.7 | 60.5 |
| **+ Mix2 (all three)** | **89.2** | **77.1** | **51.8** | **60.8** |

Key observations:
- **Individual methods underperform or match baseline** — Mixup alone actually hurts (−1.4)
- **Combination yields +4.2% overall, +6.2% on rare** vs baseline with augmentation
- **Mix2 beats ResNet152 (116M params)** with 3M params and no pretraining
- Two-method combos (Mixup + Manifold Mixup) capture most of the gain (+3.9%)

### AnuraSet-36 (59,465 segments, 36 species, no silence)

| Method | Frequent | Common | Rare | All |
|--------|----------|--------|------|-----|
| No mixing | 88.8 | 84.4 | 60.6 | 69.0 |
| Mixup only | 88.5 | 83.9 | 58.8 | 67.7 |
| **Mix2** | **89.5** | **86.8** | **67.4** | **74.2** |

Gains are consistent across dataset variants (+5.2 overall, +6.8 on rare).

### Polyphony robustness

Mix2 maintains consistent gains across all polyphony levels (1-9 simultaneous
species). Standard Mixup alone actually degrades performance at higher polyphony.
Mix2 shows very small standard deviation across runs — highly stable.

## Key insights for our implementation

### What to adopt

1. **Manifold Mixup** — highest-value single addition. Mix embeddings after the
   backbone rather than input spectrograms. Our backbone (HGNetV2-B0) outputs
   features before the LSEHead — this is the natural insertion point.

2. **Random alternation** — instead of always applying spectrogram MixUp,
   randomly choose between input-space and embedding-space mixing per batch.
   Simple to implement.

3. **Soft targets at loss level** — we already do this with our per-sample MixUp
   (lambda blending of labels). No change needed.

### What to consider

1. **MultiMix** — batch-level Dirichlet mixing. More complex to implement and
   the two-method combo (Mixup + Manifold Mixup) captures most of the gain.
   Lower priority.

2. **Selection probabilities** — 25/50/25 was empirically determined on AnuraSet.
   May need tuning for BirdCLEF. Starting with 50/50 Mixup/Manifold Mixup is
   simpler.

3. **Interaction with multi-context** — hengck23 mentions "wave mix augmentation
   specially designed for 3x5sec input." Mix2's Manifold Mixup could complement
   this by mixing at the feature level after the backbone has processed the full
   15s context.

### What differs from our setup

| Aspect | Mix2 paper | Our setup |
|--------|-----------|-----------|
| Architecture | MobileNetV3-Large (3M) | HGNetV2-B0 (6M) |
| Pretraining | None (from scratch) | ImageNet pretrained |
| Segment length | 3s | 5s |
| Sample rate | 16 kHz | 32 kHz |
| MixUp domain | Spectrogram | Spectrogram (currently) |
| Dataset size | 93K segments | 32K focal + 653 soundscape |
| Species | 42 anuran | 234 multi-taxa |
| Metric | Macro F-score | Row-wise AUROC |

## Limitations noted by authors

1. No dedicated validation set — validated on test set
2. Non-overlapping rare classes between train/test remain challenging
3. Does not incorporate ecological knowledge (temporal niche, co-occurrence priors)
4. Limited to supervised learning — self-supervised extension proposed as future work

## Implementation reference

Repository: https://github.com/ilyassmoummad/Mix2

The core implementation is straightforward:
```python
# Pseudocode for Mix2 training step
method = random.choices(['mixup', 'manifold', 'multimix'], weights=[0.25, 0.5, 0.25])[0]

if method == 'mixup':
    x_mix, y_mix = mixup(x, y, alpha)
    logits = model(x_mix)
elif method == 'manifold':
    logits, y_mix = model.forward_manifold_mixup(x, y, alpha)  # mix at random hidden layer
elif method == 'multimix':
    logits, y_mix = model.forward_multimix(x, y)  # Dirichlet batch mixing

loss = BCE(logits, y_mix)
```

For our pipeline, the key insertion point for Manifold Mixup is between the
backbone output and the LSEHead input — mix the (B, C, T') feature maps.
