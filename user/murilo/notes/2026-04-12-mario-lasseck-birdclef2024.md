---
title: "Mario Lasseck — BirdCLEF 2024 2nd Place (Pseudo-Labeling)"
source: "CEUR-WS Vol-3740/paper-199.pdf"
date: 2026-04-12
type: reference
tags: [birdclef-2024, pseudo-labeling, model-soup, iterative-training, domain-adaptation]
---

# Mario Lasseck — BirdCLEF 2024 2nd Place Solution

**Source**: CEUR-WS Vol-3740/paper-199.pdf (BirdCLEF 2024 working notes)
**Result**: 2nd place, private LB 69.035% AUC (BirdCLEF 2024)

## Key Technique: Iterative Pseudo-Labeling

Core insight: use an ensemble to generate soft pseudo-labels on 8444 unlabeled soundscapes from the target location, then mix pseudo-labeled audio with focal training data. Repeat the cycle multiple times — each iteration produces a better ensemble that generates better pseudo-labels.

### Progression (Table 2 from paper)

| Stage | Pseudo labels | Single model priv. LB | Ensemble priv. LB |
|-------|-------------|----------------------|-------------------|
| 0 | None | 59.270% | 61.738% |
| 1 | From stage 0 ensemble | 66.119% | 67.084% |
| 2 | From stage 1 ensemble | 67.445% | 69.035% |
| 3 | From stage 2 ens. (normalized) | 67.683% | 69.527% |

Single model improved +8.4% across 3 iterations. Ensemble improved +7.8%.

### How Pseudo-Labels Are Mixed Into Training

- Apply ensemble on all 5s chunks of unlabeled soundscapes → 401,947 soft pseudo-labels
- During training, randomly mix pseudo-labeled audio with focal training samples at 25-45% probability
- Before mixing audio signals, multiply amplitudes of both waveforms by a random factor: `ampFactor = 10**(random.uniform(-0.5, 0.1))` — this controls the volume ratio
- Target vector: take element-wise maximum of ground-truth labels and pseudo-label probabilities
- This is essentially noise augmentation + domain adaptation + knowledge distillation in one operation

### Three Benefits of Pseudo-Label Mixing (from paper)

1. **Noise augmentation**: model learns how species sound within environmental background noise of the test site habitat — directly addresses domain shift
2. **Training data extension**: more samples representing noise characteristics and species distribution of the deployment location
3. **Knowledge distillation**: pseudo-labels from a stronger ensemble transfer knowledge to individual models

## Baseline Architecture

Based on Salman Ahmed's public BirdCLEF 2024 notebook:

- **Backbone**: tf_efficientnet_b0_ns (ImageNet pretrained)
- **Input**: first 5 seconds of focal recordings → resized 3-channel mel spectrogram (256x256)
- **Loss**: average of BCE + Focal loss
- **Pooling**: GeM (Generalized Mean)
- **Optimizer**: AdamW with CosineAnnealingLR + 3 warmup epochs, peak lr=1e-3
- **Training**: 50 epochs with checkpoint soups (avg weights from epochs 13-50 that improve any CV metric)
- **Augmentations**: HorizontalFlip, CoarseDropout, spectrogram MixUp, local/global time+freq stretching
- **5 dropout layers** before the FC classification layer (from BirdCLEF 2021 2nd and 2023 4th place)

### Mel Parameters

- n_fft=2048, hop_length=512, n_mels=128, f_min=20, f_max=16000
- AmplitudeToDB normalization

## Ensemble Diversity Strategy (Table 4)

Mario achieves diversity by varying mel parameters, image sizes, pseudo-label mixing probability, and amplitude factors — NOT by changing backbones or architectures. All 6 models use tf_efficientnet_b0_ns.

| Model | n_mels | hop_length | Image size | PL chance | Private LB |
|-------|--------|-----------|-----------|-----------|-----------|
| 1 | 128 | 512 | 256x256 | 35% | 68.521% |
| 2 | 128 | 512 | 256x256 | 40% | 68.533% |
| 3 | 128 | 1024 | 128x128 | 45% | 68.116% |
| 4 | 64 | 1024 | 64x128 | 30% | 67.445% |
| 5 | 64 | 1024 | 64x128 | 30% | 64.543% |
| 6 | 64 | 1024 | 64x64 | 25% | 65.862% |

Key finding: just 2 models (1+2) achieved 69.694% private LB — BETTER than the full 6-model ensemble (69.035%) and better than the 1st place system (69.039%).

## Checkpoint Soups

Instead of early stopping, Mario averages weights from multiple checkpoints (epochs 13-50) that improve any validation metric. This produces more stable and occasionally better performance than picking a single best checkpoint. This is inspired by the Model Soups paper (Wortsman et al. 2022).

## Post-Processing

Simple: for each test file, predictions of each 5s window are summed with the two neighboring windows using aggregation factor 0.5 (from BirdCLEF Cornell Birdcall 3rd place, Theo Viel).

## Key Insights for Our Pipeline

1. **Pseudo-labeling is THE technique** — not recipe tweaks. Single model improved +8% across iterations.
2. **Audio mixing (not spectrogram mixing) with random amplitude factors** — addresses domain shift naturally
3. **Checkpoint soups >> single best checkpoint** — we use single best checkpoint selection
4. **Smaller, diverse models > one big model** — 64x64 mel with hop_length=1024 still contributes
5. **5 dropout layers** before FC — we use 1 dropout layer
6. **avg(BCE + Focal) loss** — not pure BCE, not pure Focal, but the AVERAGE of both
7. **Only first 5s of focal recordings used** — our random cropping may be wasting training signal on silence

## Implications for BirdCLEF 2026

- Our exp 025 pseudo-labeling attempt failed, but Mario's approach is different: he mixes pseudo-labeled AUDIO with training audio (amplitude-weighted), not just adding PL samples to the dataset
- Model soups could help stabilize our v5j — we never tested averaging checkpoints
- The "first 5s only" strategy aligns with what competitors found about beginning/end bias
- Mario's loss (avg BCE+Focal) is different from our pure BCE — worth investigating since it combines both signals
