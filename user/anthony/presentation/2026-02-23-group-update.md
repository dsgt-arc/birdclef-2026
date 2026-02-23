---
title: "BirdCLEF-2026 — Group Update"
author: Anthony Miyaguchi
date: 2026-02-23
---

## BirdCLEF 2025 Dataset at a Glance

:::::::::::::: {.columns}
::: {.column width="50%"}

| Metric | Value |
|:-------|------:|
| Total audio | **442.6 hours** |
| Focal recordings | 28,564 (280.5 h) |
| Soundscapes | 9,726 (162.1 h) |
| Species | 206 (multi-taxa) |
| Taxa | Aves, Amphibia, Insecta, Mammalia |

:::
::: {.column width="50%"}

**Competition target**: predict species at **5-second intervals** within 60-second soundscapes

- Each soundscape is exactly **60 seconds**
- 12 prediction windows per file
- Row ID: `soundscape_{id}_{end_seconds}`
- Evaluation metric: **cmAP** (class-mean average precision)

:::
::::::::::::::

## Dataset Challenges

:::::::::::::: {.columns}
::: {.column width="50%"}

**Class imbalance**:

- Top species: 9.0 hours of audio
- Bottom species: 11 seconds
- 10 species have < 45 seconds total
- Recording count ratio: **495:1**
- Audio duration ratio: **2,950:1**

:::
::: {.column width="50%"}

**Focal recording durations**:

| Percentile | Duration |
|:-----------|:--------:|
| p5 | 3.9 s |
| p25 | 10.6 s |
| Median | 21.0 s |
| p75 | 41.3 s |
| p95 | 110.9 s |

23% of recordings are < 10 seconds (need padding)

:::
::::::::::::::

## Team Experiment Landscape

| ID | Name | Owner | Status | Key Result |
|:--:|:-----|:-----:|:------:|:-----------|
| 001 | Encoder Benchmark | anthony | Complete | BEATs 95.8% ESC-50 |
| 004 | SSAMBA HF Port | anthony | Complete | Mamba ONNX custom ops |
| 007 | Pre-trained Transfer | anthony | Complete | 0.173 cmAP (2x scratch) |
| 009 | BirdCLEF Pipeline | anthony | Complete | **0.484 Kaggle public** |
| 010 | Teacher Labels | anthony | Complete | 88 GB labels, 3 models |
| 012 | EfficientNet Probe | anthony | Complete | 0.315 cmAP frozen |
| 001 | BirdSet Benchmark | murilo | Complete | ConvNeXt 0.48 cmAP |
| 003 | Model Training | murilo | Complete | ConvNeXT-T 0.180 cmAP |

## Backbone Quality is the Bottleneck

| Model | Pre-training | Params | cmAP | Method |
|:------|:-------------|:------:|:----:|:------:|
| Bird-MAE ViT-L | Bird audio | 300M+ | 0.55 | SOTA |
| ConvNeXt (BirdSet) | Audio | ~30M | 0.48 | murilo/001 |
| **EfficientNet-B1** | **BirdSet-XCL** | **264K trainable** | **0.315** | **Linear probe** |
| ConvNeXT-Tiny | ImageNet | 27.8M | 0.180 | murilo/003 |
| SSAMBA Tiny | AudioSet | 7M | 0.173 | Full finetune |
| SSAMBA Tiny | None | 7M | 0.086 | Full finetune |

EfficientNet-B1 with **3.9% trainable params** beats SSAMBA with 100% trainable params by 76%

## End-to-End Pipeline (Exp 009)

:::::::::::::: {.columns}
::: {.column width="50%"}

**Training pipeline**:

- SSL pretrain (MPG span masking)
- Finetune with frozen→unfrozen schedule
- AudioSet backbone init: **0.425 val/cmap**
- SpecAugment, focal BCE, 206 species

:::
::: {.column width="50%"}

**Kaggle submission**:

| Metric | Value |
|:-------|------:|
| Public score | **0.484** |
| Private score | **0.476** |
| Runtime | 13 seconds |
| Model size | 27.3 MB (ONNX) |
| GPU required | No |

CPU-only inference via custom `mamba-onnx` ops

:::
::::::::::::::

## Teacher Labels Pipeline (Exp 010)

Three independent teacher models ran on **all 442 hours** of BirdCLEF 2025 audio:

| Teacher | Approach | Species | Focal Clips | Soundscape Clips |
|:--------|:---------|:-------:|:-----------:|:----------------:|
| **BamBird** | Unsupervised ROI + wavelet features | 201 | 660K ROIs | — |
| **BirdNET** | Neural classifier (3s/1s step) | 6,522 | 939K | 564K |
| **Perch** | TFLite embedding model (5s step) | 10,932 | 188K | 117K |

**Total output**: 88 GB of predictions, embeddings, and features

## BamBird: Unsupervised Event Detection

:::::::::::::: {.columns}
::: {.column width="50%"}

**ROI extraction** (bandpass 250–12kHz):

| Metric | Value |
|:-------|------:|
| Files processed | 28,564 |
| Files with ROIs | 21,814 (76%) |
| Total ROIs | **660,081** |
| ROIs/file (mean) | 30.3 |
| ROIs/file (median) | 5.0 |
| Event duration (mean) | 0.80 s |
| Event duration (median) | 0.44 s |

:::
::: {.column width="50%"}

**DBSCAN clustering** (signal vs noise):

| Metric | Value |
|:-------|------:|
| Signal ROIs | 81,537 (12.4%) |
| Noise ROIs | 578,544 (87.6%) |
| Files with 0 signal | 12,149 (55.7%) |
| Species with 0% signal | 56/201 (27.9%) |

BamBird uses 96-dim wavelet features and per-species DBSCAN. Quality issues: auto-tuned `min_points` too aggressive.

:::
::::::::::::::

## Soundscapes Are Mostly Quiet

Fraction of clips where max species confidence > 0.50:

:::::::::::::: {.columns}
::: {.column width="50%"}

**Focal recordings** (expected: lots of bird audio)

| Teacher | Active Clips | % |
|:--------|:------------|---:|
| BirdNET | 427,716 | 45.5% |
| Perch | 120,981 | 64.4% |

:::
::: {.column width="50%"}

**Soundscapes** (wild field recordings)

| Teacher | Active Clips | % |
|:--------|:------------|---:|
| BirdNET | 30,393 | **5.4%** |
| Perch | 12,992 | **11.1%** |

:::
::::::::::::::

Only **5–11% of soundscape clips** have confident bird detections. The competition test set is entirely soundscapes — models must handle sparse, noisy audio.

## Why Weak Supervision?

:::::::::::::: {.columns}
::: {.column width="50%"}

**The problem**:

- Focal recordings: single species label for entire clip
- No per-frame annotations
- Soundscapes: multi-species, no labels at all
- Extreme class imbalance (2,950:1 in duration)

:::
::: {.column width="50%"}

**Three teachers, three error modes**:

- **BamBird**: unsupervised wavelet features — detects acoustic events regardless of species
- **BirdNET**: neural classifier trained on xeno-canto
- **Perch**: Google's embedding model, independent architecture

Independent errors → ideal for **FlyingSquid** label fusion (triplet method)

:::
::::::::::::::

## 2025 Competition Landscape

:::::::::::::: {.columns}
::: {.column width="50%"}

**What top solutions used** (>0.85 score):

- SED + AttBlockV2 (+0.03–0.05)
- Iterative pseudo-labeling (3–4 rounds)
- Standard CNNs (EfficientNet, RegNet, NFNet)
- 7-model ensembles at top

:::
::: {.column width="50%"}

**Key insight from 1st place**:

> Pseudo-labeling was the dominant factor, not architecture

- Temperature 0.55–0.6
- Temporal smoothing `[0.2, 0.6, 0.2]`
- Train with attention pooling, infer with framewise max

:::
::::::::::::::

Our teacher labels pipeline is a step toward the same idea — systematic pseudo-labels from multiple sources

## Where We Stand

| Approach | Kaggle Public | Gap to 1st (0.85+) |
|:---------|:------------:|:-------------------:|
| SSAMBA Tiny (ours, CPU) | 0.484 | -0.37 |
| EfficientNet baseline | 0.778 | -0.07 |
| BirdNET baseline | ~0.72 | -0.13 |

**Key findings**:

- Backbone quality >> pooling strategy >> model size
- Pre-training is essential — scratch models plateau at 0.02–0.09 cmAP
- Domain-specific pre-training (BirdSet-XCL) >> generic (ImageNet, AudioSet)
- 88 GB of teacher labels ready for weak supervision fusion

## Next Steps

1. **Weak supervision fusion** — FlyingSquid with BamBird + BirdNET + Perch as 3 labeling functions
2. **SSL finetune comparison** — does BirdCLEF-pretrained SSAMBA beat AudioSet init?
3. **EfficientNet full finetune** — unfreeze backbone to push past 0.315 linear probe
4. **Pseudo-labeling pipeline** — use teacher labels to iteratively refine training data
5. **Kaggle resubmission** — integrate best backbone + teacher-guided training
