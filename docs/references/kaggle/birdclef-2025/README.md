# BirdCLEF 2025 Kaggle Notebook Index

Clean-room summaries of the top 20 public notebooks by votes.

## Scored Notebooks (by LB)

| LB | Votes | Summary | Architecture | Key Technique |
|----|-------|---------|-------------|---------------|
| 0.878 | 577 | [WeightedBlend 0.88](bird25-weightedblend-0-88.md) | NFNet + SEResNeXt (75/25) | Power adjustment, temporal smoothing |
| 0.878 | 241 | [NFNet+SEResNeXt Blend](bird25-weightedblend-nfnet-seresnext-0-878.md) | NFNet + SEResNeXt (75/25) | 10s context, temporal TTA |
| 0.874 | 375 | [OpenVINO Ensemble](bird25-openvino-ensemble-infer-baseline-lb-874.md) | SED + OpenVINO + NFNet (50/35/15) | 3-stage ensemble, conservative tuning |
| 0.860 | 436 | [NFNet+ConvNeXtV2 Blend](bird25-nfnet-convnextv2-weightedblend-lb-860.md) | NFNet + ConvNeXtV2-Nano (60/40) | Power adjustment, diverse architectures |
| 0.857 | 329 | [Single SED Model](bird2025-single-sed-model-inference-lb-0-857.md) | SEResNeXt26t + AttBlockV2 | 10s context, framewise max pooling |
| 0.834 | 363 | [OnlyIN Changed CFG](bird25-onlyin-lb-834-changed-cfg.md) | EfficientNetV2-S | FocalLossBCE, config tuning |
| 0.829 | 401 | [FocalLossBCE CV.962](bird25-onlyinf-v2-s-focallossbce-cv-962-lb-829.md) | EfficientNetV2-S | FocalLossBCE (0.6 BCE + 1.4 focal) |
| 0.810 | 138 | [EfficientNetV2-B0](lb-0-81-efficientnetv2-b0-pytorch-inference.md) | EfficientNetV2-B0 | TTA, fold ensemble |
| 0.804 | 385 | [LB 0.804 Pipeline](lb-0-804-efficientnet-b0-pytorch-pipeline.md) | EfficientNet-B0 | Optimized mel params, temporal smoothing |
| 0.784 | 159 | [Labels TTA](labels-tta-efficientnet-b0-pytorch-inference.md) | EfficientNet-B0 | TTA (h/v flip), temporal smoothing |

## Training / Baseline Notebooks

| Votes | Summary | Purpose |
|-------|---------|---------|
| 936 | [EfficientNet B0 Train](efficientnet-b0-pytorch-train-birdclef-25.md) | Full training pipeline: 5-fold CV, mixup, SpecAugment |
| 710 | [EfficientNet B0 Inference](efficientnet-b0-pytorch-inference-birdclef-25.md) | Companion inference notebook |
| 269 | [RegNetY008 v1 Submission](bird-25-submission-regnety008-v1.md) | Lightweight RegNet backbone |
| 155 | [RegNetY008 Default](bird-25-regnety008-v1-default-v1.md) | RegNet variant with defaults |
| 245 | [Sample Submission](birdclef-2025-sample-submission.md) | Official baseline template (Stefan Kahl) |

## Utility / EDA Notebooks

| Votes | Summary | Purpose |
|-------|---------|---------|
| 530 | [Sound Event Visualisations](birdclef2025-sound-event-visualisations.md) | Mel + prediction heatmap debugging |
| 450 | [Audio-to-Mel Spec](transforming-audio-to-mel-spec-birdclef-25.md) | Precompute spectrogram pipeline |
| 351 | [Post-Processing Power Adjustment](post-processing-with-power-adjustment-for-low-rank.md) | Power adjustment technique deep-dive |
| 181 | [Voice Separation](bc25-separation-voice-from-data.md) | Silero VAD for speech contamination |
| 146 | [MFCC ROC-AUC Analysis](birdclef-2025-mfcc-feature-roc-auc-analysis.md) | Traditional ML baseline (XGBoost/LightGBM) |

## Landscape Summary

### Score Tiers

- **0.87-0.88**: Multi-architecture SED ensembles (NFNet + SEResNeXt) with power adjustment
- **0.85-0.86**: Single SED model with AttBlockV2 attention pooling and 10s context windows
- **0.82-0.84**: EfficientNetV2-S with FocalLossBCE — the loss function upgrade alone is worth +0.03
- **0.78-0.81**: EfficientNet-B0/V2-B0 image classification baseline with TTA

### Universal Techniques

| Technique | Prevalence | Typical Gain |
|-----------|-----------|-------------|
| Temporal smoothing [0.2, 0.6, 0.2] | ~15/20 notebooks | +0.01-0.02 |
| TTA (time/freq flip) | ~12/20 notebooks | +0.01 |
| Power adjustment (top_k=30, exp=2) | 5/20 notebooks (all top-5) | +0.01-0.02 |
| FocalLossBCE | 3/20 notebooks | +0.02-0.03 |
| SED + AttBlockV2 | 5/20 notebooks (all top-5) | +0.03-0.05 |
| Cross-architecture ensemble | 4/20 notebooks (top-4) | +0.02 |

### Consensus Mel Parameters

32 kHz sample rate, 1024 n_fft, 512 hop_length, 128 mel bins, 50-14000 Hz, dB + min-max normalization, 256x256 resize. SED models sometimes use 256 hop_length and native resolution with mean/std normalization.

### Dominant Backbones

1. **eca_nfnet_l0** — appears in 4 of top 5 ensembles
2. **seresnext26t_32x4d** — the go-to SED backbone, paired with NFNet
3. **EfficientNetV2-S** (in21k_ft_in1k) — best single-architecture results
4. **ConvNeXtV2-Nano** — diversity partner in one top ensemble
5. **RegNetY-008** — lightweight alternative (~6M params)
