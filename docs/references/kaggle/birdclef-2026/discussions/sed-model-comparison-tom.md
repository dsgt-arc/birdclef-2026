---
title: "SED Model Comparison Report — 28 Experiments (Tom)"
source: "https://storage.googleapis.com/kaggle-forum-message-attachments/3424971/39433/sed_model_comparison_20260320.html"
kaggle_discussion: "https://www.kaggle.com/competitions/birdclef-2026/discussion/681146"
date_captured: "2026-03-25"
date_report: "2026-03-20"
related: ["sed-improvement-plan-681146.md", "../../winning-solutions-analysis.md"]
---

# SED Model Comparison Report — 28 Experiments

Comprehensive ablation study by competitor "Tom" on BirdCLEF 2026 SED architectures.
28 experiments testing backbones, loss functions, augmentation, knowledge distillation,
and pseudo-labeling strategies. Best model: **0.8139 soundscape val AUC**.

## Baseline Architecture (All Experiments)

| Parameter | Value |
|-----------|-------|
| Backbone | `tf_efficientnet_b0.ns_jft_in1k` (4.0M params, 1280-dim) |
| Sample rate | 32,000 Hz |
| Clip duration | 5 seconds |
| n_fft | 2048 |
| hop_length | 512 |
| n_mels | 224 |
| fmin / fmax | 0 / 16,000 Hz |
| Normalization | AmplitudeToDB (top_db=80) + per-sample min-max |
| Head | GEMFreqPool (p=3.0, learnable) + tanh attention + softmax temporal pooling |

## Experiment Results

### Phase 1: Initial Baselines

| Experiment | Loss | Technique | SS Val AUC | Holdout AUC | Best Ep |
|------------|------|-----------|-----------|-------------|---------|
| sed-b0-v5 | BCE | Baseline dual BCE (clip+frame) | 0.7527 | 0.9192 | 7 |
| sed-b0-v6 | BCE | 30 epochs | 0.7163 | — | 21 |
| sed-b0-v10-cutmix | BCE | CutMix augmentation | 0.7060 | — | 2 |
| sed-b0-v11-soft-sec | BCE | Soft secondary labels | 0.7098 | — | 3 |
| sed-b0-v12-bce | BCE | Plain BCE (no dual) | 0.7263 | — | 2 |
| sed-b0-v16-rating3 | BCE | Rating ≥3 filter | 0.6250 | — | 2 |

### Phase 2: Asymmetric Loss (ASL)

| Experiment | Loss | Technique | SS Val AUC | Holdout AUC | Best Ep |
|------------|------|-----------|-----------|-------------|---------|
| sed-b0-v9-asl | ASL | ASL + pseudo r1, 20ep soup | 0.7755 | 0.9467 | 20 |
| sed-b0-v13-asl-cutmix | ASL | ASL + CutMix | 0.7467 | — | 6 |
| sed-b0-v15-no-sec | ASL | ASL no secondary labels | 0.6842 | — | 1 |
| sed-b0-v26-asl-npcen | ASL | ASL + NPCEN frontend | 0.7262 | 0.9574 | 3 |

### Phase 3: Dual Loss Ablations

| Experiment | Loss | Technique | SS Val AUC | Holdout AUC | Best Ep |
|------------|------|-----------|-----------|-------------|---------|
| sed-b0-v21-dual-rating3 | Dual BCE | Dual (0.5 clip + 0.5 frame), rating≥3 | 0.7289 | — | 13 |
| sed-b0-v22-dual-noclipmix | Dual BCE | Dual without ClipMix | 0.7485 | 0.8658 | 5 |

### Phase 4: Knowledge Distillation from Perch

| Experiment | Loss | Technique | SS Val AUC | Holdout AUC | Best Ep |
|------------|------|-----------|-----------|-------------|---------|
| sed-b0-v23-perch-asl-aug | ASL | Perch augmentation KD | 0.7367 | — | 4 |
| sed-b0-v24-soft-pseudo | ASL+softKD | Soft pseudo-label KD (early) | **0.8141** | 0.9199 | 12 |
| sed-b0-v27-soft-boost | ASL+softKD | Boosted soft pseudo weight | 0.7804 | 0.9915 | 10 |

### Phase 5: Multi-round Pseudo Labels

| Experiment | Loss | Technique | SS Val AUC | Holdout AUC | Best Ep |
|------------|------|-----------|-----------|-------------|---------|
| sed-b0-v28-final-combo | ASL+softKD | r4 pseudo + soft KD 10x + SpecAugment | 0.8053 | 0.9861 | 9 |
| sed-b0-v30-bgnoise | ASL+softKD | r1-r5 pseudo + bg noise aug | 0.7725 | 0.9627 | 4 |
| **sed-b0-v30-multipseu** | **ASL+softKD** | **r1-r5 pseudo, soft KD 10x, no bg noise** | **0.8139** | **0.9839** | **6** |

### Phase 6: 4-Fold GroupKFold (In Progress)

| Experiment | Loss | Technique | SS Val AUC | Best Ep | Status |
|------------|------|-----------|-----------|---------|--------|
| sed-b0-4fold-v30-fold0 | ASL+softKD | GroupKFold fold 0 | 0.8033 | 10 | running |
| sed-b0-4fold-v30-fold1 | ASL+softKD | GroupKFold fold 1 | 0.7952 | 3 | early stop |
| sed-b0-4fold-v30-fold2 | ASL+softKD | GroupKFold fold 2 | 0.7656 | 4 | running |

### Alternative Backbones

| Experiment | Backbone | Params | SS Val AUC | Holdout AUC | Notes |
|------------|----------|--------|-----------|-------------|-------|
| sed-b2-v1 | B2 | 9.1M | 0.7104 | — | Underperforms B0 |
| sed-b3-v1-asl | B3-NS | 12M | 0.7805 | 0.9553 | Oscillates after ep2 |
| sed-v2s-v1 | EfficientNetV2-S | 21M | 0.7012 | 0.7976 | Poor transfer |
| sed-v2s-v3 | EfficientNetV2-S | 21M | 0.7592 | — | Still unstable |

## Cumulative Technique Impact

| Step | Change | SS Val AUC | Delta |
|------|--------|-----------|-------|
| Baseline | BCE dual loss | 0.7527 | — |
| + ASL | gamma_neg=4, gamma_pos=0, clip=0.05 | 0.7755 | **+0.023** |
| + Soft KD | Perch teacher, frame_weight=0.5 | 0.7804 | **+0.005** |
| + Multi-round pseudo + KD 10x | r4 pseudo labels | 0.8053 | **+0.025** |
| + r1-r5 pseudo (no bg noise) | Multi-round refinement | 0.8139 | **+0.009** |
| **Total** | | **0.8139** | **+0.061** |

## What Works

- **ASL loss (+0.023)**: Asymmetric focusing suppresses easy negatives in 234-class setting
- **Soft KD from Perch (+0.030 cumulative)**: Bridges focal→soundscape domain gap
- **Multi-round pseudo labels (+0.008)**: 5 rounds of Perch-generated soft labels
- **ClipMix in dual loss**: Critical dependency — removing it drops AUC significantly
- **NS-JFT pretrained weights**: `tf_efficientnet_b0.ns_jft_in1k` outperforms standard ImageNet
- **Dual loss (clip + frame)**: Better than plain BCE
- **AmplitudeToDB + min-max normalization**: Used across all experiments

## What Doesn't Work

- **Background noise augmentation (−0.041)**: Mismatch with pseudo labels generated on clean audio
- **Soft secondary labels (−0.043)**: Conflicting co-occurrence gradients
- **Rating filter ≥3 (−0.128)**: Data quantity dominates quality for 234 classes
- **EfficientNetV2-S (−0.043 vs B0)**: ImageNet-21k pretraining may overfit to visual features
- **EfficientNet-B3**: Unstable training, oscillates after warmup (LR=5e-4 too high for 1536-dim)
- **CutMix (−0.047)**: Destroys spatial coherence of spectrograms
- **NPCEN frontend**: Better holdout but worse soundscape val (overfits train_audio distribution)

## Key Insights

1. **Soundscape val AUC is the real metric.** Holdout AUC has +0.14 to +0.22 gap to soundscape val.
   Models that excel on holdout often fail on soundscapes (domain shift).

2. **B0 is the sweet spot.** B2, B3, V2S all underperform or are unstable.
   B3 needs careful LR tuning (Tom's LR=5e-4 was too high).

3. **Perch KD is a major lever.** Soft labels from Perch on soundscape data bridge the
   domain gap that focal-only training cannot close.

4. **Don't filter training data by quality.** Use everything — rare species need all
   available samples.

## Relevance to Our Exp 017

| Tom's finding | Our status | Action |
|---------------|-----------|--------|
| n_mels=224 | Testing in v4a | Already aligned |
| AmplitudeToDB + min-max | Planned Step 2 (v4b) | Validates this direction |
| tanh+softmax head | Planned Step 2 (v4c) | Validates AttBlockV2 |
| ASL loss (+0.023) | Planned Step 3 (v4f) | Strong signal, priority |
| NS-JFT weights | Our v2 regressed (−0.068) | May need ASL to work (interaction) |
| B3 unstable | Our v3d regressed | Confirms B0 is correct choice |
| EfficientNetV2 poor | Our v3d regressed | Confirms V1 architecture |
| Perch soft KD (+0.030) | Not planned yet | **New opportunity for exp 017/018** |
| Multi-round pseudo | Planned exp 018 | Validates approach |
| BG noise augmentation hurts | Not planned | Avoid this |

Note: Tom uses n_fft=2048 (not 4096 like 2025 1st place) and still gets strong results
with 224 mels. This suggests n_mels=224 may matter more than n_fft=4096.
