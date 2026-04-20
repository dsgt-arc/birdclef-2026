# Project Status

> **Last Updated**: 2026-04-20

## Current State

**Best LB**: **0.918** (Perch ProtoSSM v4, murilo/exp 018 — on `mg/018-perch-ensemble` branch)
**Best SED LB**: **0.859** (EfficientNet-B0 SED v4b, murilo/exp 017 — on branch)
**Anthony's best**: **0.804** (exp 106 ridge on frozen Perch v2, shipped via exp 108 scaffold 2026-04-20)
**Target**: 0.925+ (match top public notebooks), then 0.939+ (competition #1)

### Score Progression

| Model | Experiment | LB | Notes |
|-------|-----------|-----|-------|
| EfficientNet-B1 frozen probe | anthony/012 | 0.649 | Frozen backbone + linear head |
| ProtoCLR clip-level | murilo/012 | 0.767 | Best clip-level model |
| **EfficientNet-B1 LoRA + attention** | **anthony/102** | **0.794** | 2 epochs, attention pooling critical |
| **Perch v2 ridge (frozen)** | **anthony/106 via 108** | **0.804** | Ridge on frozen Perch v2 embeddings; LOSO 0.749 conservative by +0.055 |
| EfficientNet-B0 SED v1 + pp2 | murilo/016 | 0.827 | SED architecture + post-processing |
| Perch raw (no probes) | murilo/018 | 0.825 | Foundation model baseline |
| Perch + genus proxy | murilo/018 | 0.836 | 6 genus-mapped species |
| **EfficientNet-B0 SED v4b** | **murilo/017** | **0.859** | **Mel overhaul + AmplitudeToDB + AttHead** |
| Perch + LogReg probes | murilo/018 | 0.893 | PCA→32d, 39-dim features |
| Perch + OOF v2-fix | murilo/018 | 0.911 | 5-fold GroupKFold, bug fixes |
| **Perch ProtoSSM v4** | **murilo/018** | **0.918** | **BiSSM temporal + cross-attn + gated fusion** |
| Competitor: Yuriy ProtoSSM v5 | — | 0.925 | Top public notebook |
| Competition #1 | — | 0.939 | Likely ensemble + pseudo-labels |

> Exp 017, 018, 019 results are on the `mg/018-perch-ensemble` branch (not yet merged to main).

### Key Findings

**From Exp 017 (Murilo, on branch):**
1. **Mel resolution is the biggest lever** — n_fft 1024→4096, n_mels 128→224 gave +0.030 LB alone
2. **AmplitudeToDB universal** — +0.034 LB on top of mel changes (v4b = 0.859 LB)
3. **Techniques don't stack additively** — v4f/v4g/v4h all regressed vs v4b
4. **Fulldata training regressed** — −0.006 LB, overfitting without epoch caps
5. **Offline metrics are poor LB predictors** — v4e best val (0.868) but only 0.840 LB

**From Exp 018 (Murilo, on branch):**
1. **Perch-only path dominates 0.910+** — every top public notebook uses frozen Perch v2
2. **OOF infrastructure = +0.018** — proper fold-safe priors, filename grouping, config bug fixes
3. **ProtoSSM temporal modeling = +0.007** — BiSSM + cross-attention + gated fusion on top of OOF probes
4. **Batch fixes are risky** — v4.2 changed 4 things at once, regressed −0.019

**From Exp 019 (Murilo, on branch):**
1. **v5a (BCE loss) improved val +0.010 but regressed LB −0.009** — offline/LB divergence continues
2. **Phase A ablations inconclusive** — no single change beat SED v4b (0.859) on LB

**From Exp 102 (Anthony):**
1. **Attention pooling is critical** — 0.794 LB vs 0.754 for avg pooling
2. **Focal overfitting is the main problem** — 2-epoch model > 13-epoch model on LB
3. **Soundscape validation did NOT predict Kaggle LB** — SC auroc 0.870 → 0.528 LB
4. **Teacher predictions**: 5.4% noise, 29% potential mislabels

## Murilo's Experiments

### On `main` (merged)

| ID | Name | Status | Result |
|----|------|--------|--------|
| 001 | BirdSet Benchmark | Complete | ConvNeXT 0.48, EfficientNet 0.38 on HSN |
| 002 | Supervised ViT Baseline | Complete | 0.186 cmAP — ImageNet pretraining insufficient for audio |
| 003 | Model Training | Complete | ConvNeXT-Tiny pretrained best (cmAP=0.191), pretraining 2.2x boost |
| 004 | Frame-Based Tokenization | Complete | 128×4 vertical tokens: cmAP=0.118 (+37%), severe overfitting |
| 005 | Pretrained Audio Fine-Tuning | Complete | EAT frozen probe: AUROC=0.890, cmAP=0.301 |
| 006 | LoRA Fine-Tuning EAT | Complete | AUROC=0.945, cmAP=0.536 with 0.67% trainable params |
| 007 | ConvNeXt-Base LoRA | Complete | AUROC=0.941, ImageNet nearly matches AudioSet pretraining |
| 008 | Frame Tokenization for EAT | Complete | Rejected — pretrained patch embeddings essential |
| 009 | ProtoCLR LoRA Fine-Tuning | Complete | AUROC=0.942, 19M params matches 90M models |
| 010 | Submission Pipeline | Complete | ONNX export + Kaggle kernel, 3 architectures |
| 011 | LoRA Baseline 2026 | Complete | ProtoCLR fulldata 0.736 LB |
| 012 | Soundscape Fine-Tuning | Complete | ProtoCLR **0.767 LB** (+0.031) |
| 013 | MixIT Source Separation | Complete | 10,592 soundscapes → 4 channels, pipeline ready |
| 014 | Teacher Pseudo-Labeling | Complete | 127K segments, ensemble AUROC 0.885 |
| 015 | Student Training | **Rejected** | Pseudo-labels hurt with clip-level arch (−0.005 to −0.024) |
| 016 | SED Architecture | Complete | **0.827 LB** (v1 + pp2 + 10s inference) |

### On `mg/018-perch-ensemble` branch (not merged)

| ID | Name | Status | Result |
|----|------|--------|--------|
| 017 | Mel Overhaul | Complete | **0.859 LB** (v4b: AmplitudeToDB + AttHead + soft CE) |
| 018 | Perch Probes / ProtoSSM | In Progress | **0.918 LB** (v4), v4.2 regressed to 0.899 |
| 019 | Perch-Guided SED Training | Phase A done | Ablations inconclusive, offline≠LB gap |
| 020 | Perch+SED Ensemble | Proposed | Not started |
| 021 | Advanced Post-Processing | Proposed | Not started |

### Architecture Evolution

```
Vision pretrain (001-004)     Audio pretrain + LoRA (005-009)     2026 Pipeline (010-012)
  ViT/ConvNeXT scratch          EAT/ProtoCLR/ConvNeXt LoRA         Submission + soundscape
  cmAP ~0.12-0.19               AUROC 0.94, cmAP 0.53              0.767 LB (clip-level)
         ↓                              ↓                                  ↓
Clip-level bottleneck (013-015)    SED + Mel (016-017)           Perch Probes (018-019)
  Pseudo-labels failed              0.859 LB (SED v4b)            0.918 LB (ProtoSSM v4)
  GlobalAvgPool = ceiling           Mel resolution key             Frozen Perch + temporal
```

## Anthony's Experiments

| ID | Name | Status | Result |
|----|------|--------|--------|
| 001 | Encoder Benchmark | Complete | BEATs 95.8%, EAT+LoRA 94.6% |
| 009 | BirdCLEF End-to-End Pipeline | Paused | SSL pretrain done, finetune pending |
| 010 | Teacher Labels | Complete | BirdNET + Perch + BamBird (88 GB) |
| 011 | Prompted SED | Phase 0 Done | Inconclusive — emerging signal |
| 012 | EfficientNet-B1 Linear Probe | Complete | 0.649 Kaggle (frozen backbone) |
| 013 | Weak Supervision (FlyingSquid) | Complete | Soft labels generated |
| 014 | EfficientNet + Soft Labels | Rejected | Soft labels hurt (−0.098 cmAP) |
| 015 | EfficientNet SED + Soft Labels | Rejected | SED needs trainable backbone |
| 016 | Contrastive SED / Prototypical | Complete | Step 0 done |
| 100 | BirdCLEF 2026 EDA | Complete | 206 species, amphibian-dominated, 499x imbalance |
| 101 | Soundscape Label Calibration | Complete | Ensemble 0.880 AUROC, spectral contrast 0.888 |
| 102 | Multi-Backbone LoRA Ladder | Paused | 0.794 Kaggle LB, attention pooling, teacher filtering |
| 106 | Perch v2 Bayesian Label Transfer | Complete | Ridge **0.804 Kaggle LB** (+0.010 over 102), LOSO 0.749 conservative |
| 107 | Hopfield Prototype Retrieval | In Progress | Null-anchor S18 0.7801 pooled LOSO (v4 0.7074); LB pending |
| 108 | Kaggle Submission Scaffold | **In Progress** | Shared `birdclef_kaggle` module; ridge shipped; hopfield null-anchor next |

## Concept Docs

| Concept | Key Idea | Status |
|---------|----------|--------|
| [Strong Baseline Mel Overhaul](concepts/strong-baseline-mel-overhaul.md) | Close SED gap via mel resolution + recipe alignment | Done (exp 017) |
| [Prompted SED](concepts/prompted-sound-event-detection.md) | Reference audio as Mamba prefix for few-shot SED | Developing — phase 0 inconclusive |
| [Source Separation Training](concepts/source-separation-training.md) | Train PyTorch MixIT separator on BirdCLEF data | Developing — concept ready, not started |
| [Source Separation Augmentation](concepts/source-separation-augmentation.md) | Biodenoising + MixIT for smart augmentation | Developing — 4 strategies, 5 hypotheses |
| [Weak Supervision Label Fusion](concepts/weak-supervision-label-fusion.md) | FlyingSquid triplet method to fuse noisy teacher signals | Done (exp 013) |
| [Soundscape Domain Shift](concepts/soundscape-domain-shift.md) | XenoCanto → soundscape gap analysis | Fundamental challenge |
| [SSAMBA Audio Mamba](concepts/ssamba-audio-mamba.md) | Mamba backbone for Track 2 | Mature — validated through exps 004-009 |

## Competition Intelligence

| Resource | Location |
|----------|----------|
| SED improvement plan (28-experiment ablation) | [discussions/sed-improvement-plan-681146.md](references/kaggle/birdclef-2026/discussions/sed-improvement-plan-681146.md) |
| Perch + ProtoSSM (0.913 LB) | [notebooks/perch-protossm-zirach.md](references/kaggle/birdclef-2026/notebooks/perch-protossm-zirach.md) |
| Pantanal Distill (0.917 LB) | [notebooks/pantanal-distill.md](references/kaggle/birdclef-2026/notebooks/pantanal-distill.md) |
| Perch + LogReg (0.910 LB) | [notebooks/perch-probes-chiranjithdharma.md](references/kaggle/birdclef-2026/notebooks/perch-probes-chiranjithdharma.md) |
| 2025 1st place inference | [birdclef-2025/birdclef2025-1st-place-inference.md](references/kaggle/birdclef-2025/birdclef2025-1st-place-inference.md) |
| Winning solutions analysis | [winning-solutions-analysis.md](references/kaggle/winning-solutions-analysis.md) |
| Gemini deep research | [deep-research/2026-03-24-gemini-strategic-framework.md](references/deep-research/2026-03-24-gemini-strategic-framework.md) |
| ttahara HGNetV2-B0 (0.884 LB) | Referenced in murilo/exp 019 |
| Yuriy ProtoSSM v5 (0.925 LB) | Referenced in murilo/exp 018 |

## Priority Stack

1. **Close Perch ProtoSSM gap** — 0.918 → 0.925 (revert v4.2 regression, debug remaining 0.007)
2. **Perch-guided pseudo-labeling** — use ProtoSSM teacher on 10,600 unlabeled soundscapes for SED training
3. **SED ablation debugging** — understand offline/LB divergence before more ablations
4. **Perch + SED ensemble** — combine 0.918 Perch + 0.859 SED for potential gains
5. **Anthony exp 107 null-anchor LB** — ship `softshrink_null_site_anchored_residual` S18 (0.7801 pooled LOSO) via exp 108; characterize ridge→hopfield LB delta
6. **Anthony exp 102** — push LoRA+attention past 0.794 (teacher filtering, more epochs with early stop)

## Quick Links

| Resource | Link |
|----------|------|
| Concepts | [concepts/](concepts/) |
| Competition Intel | [references/kaggle/](references/kaggle/) |
| Deep Research | [references/deep-research/](references/deep-research/) |
| Literature | [references/literature.md](references/literature.md) |
| Vendor Docs | [vendor/](vendor/) |
| Templates | [_templates/](_templates/) |
| Strategy | [STRATEGY.md](STRATEGY.md) |
