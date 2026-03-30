# Project Status

> **Last Updated**: 2026-03-30

## Current State

**Best LB**: **0.827** (EfficientNet-B0 SED v1 + pp2 + 10s inference, murilo/exp 016)
**Anthony's best**: **0.794** (EfficientNet-B1 LoRA + attention pooling, exp 102 run C)
**Target**: 0.85+ labeled-data baseline, then 0.90+ with pseudo-labeling

### Score Progression

| Model | Experiment | LB | Notes |
|-------|-----------|-----|-------|
| EfficientNet-B1 frozen probe | anthony/012 | 0.649 | Frozen backbone + linear head |
| ProtoCLR clip-level | murilo/012 | 0.767 | Best clip-level model |
| **EfficientNet-B1 LoRA + attention** | **anthony/102** | **0.794** | **2 epochs, attention pooling critical** |
| EfficientNet-B1 attention + SC val | anthony/102 | 0.609 | SC validation didn't predict LB |
| EfficientNet-B1 attention + teacher filter + SC val | anthony/102 | 0.528 | Teacher filtering + SC val hurt LB |
| EfficientNet-B0 SED v1 | murilo/016 | 0.795 | SED architecture validated |
| + post-processing (TTA + smoothing + TopN) | murilo/016 | 0.823 | +0.028 free gains |
| **+ pp2 (temp scaling + taxa smoothing + priors)** | **murilo/016** | **0.827** | **Overall best** |
| Competitor SED (single model) | — | 0.908 | Our target |
| Perch + ProtoSSM | — | 0.913 | Foundation model approach |
| Pantanal Distill | — | 0.917 | Best public notebook |

### Key Findings

**From Exp 102 (Anthony):**
1. **Attention pooling is critical for soundscape performance** — 0.794 LB vs 0.754 for avg pooling (framewise max catches brief calls)
2. **Focal overfitting is the main problem** — more training epochs improves val/cmap but hurts Kaggle LB (2-epoch model > 13-epoch model on LB)
3. **ConvNeXt-Base overfits worse than EfficientNet-B1** — 0.644 val/cmap but only 0.603 Kaggle LB (89M params + 128 mels)
4. **Soundscape validation did NOT predict Kaggle LB** — SC auroc 0.870 → 0.528 LB. The 50/50 site split may cause train/val site mismatch. The 2-epoch model (0.794 LB) trained without any soundscape validation remains best.
5. **bf16 emulated on RTX 6000 (sm_75)** — fp16 gives 3x throughput; torch.compile works with 2.11+cu126
6. **Teacher predictions**: 5.4% of focal chunks are noise (species not present), 29% potential mislabels

**From Exp 016 (Murilo):**
1. **SED +0.112 cmAP over clip-level** — framewise detection captures brief calls
2. **Every v3 ablation regressed vs v1** — gap is NOT in weights, loss, or mixup domain
3. **Mel resolution gap**: 1024 vs 4096 n_fft is the largest untested difference vs competitors
4. **Post-processing is free gains** — TTA + smoothing + TopN + taxa priors adds +0.032

### What's Next

**Pseudo-labeling soundscapes** — run trained attention model on 10,592 unlabeled soundscape files. Key design questions: threshold vs ranking vs soft labels for uncalibrated probabilities. Design doc in progress.

**Exp 017: Strong Baseline via Mel Overhaul** — close the gap to 0.908 using mel resolution (n_fft=4096/224 mels). See [concept doc](concepts/strong-baseline-mel-overhaul.md).

## Key Findings

| Finding | Source | Implication |
|---------|--------|-------------|
| **SED v1 = 0.827 LB** (best single model) | murilo/Exp 016 | SED + pp2 + 10s inference |
| **EfficientNet LoRA+attention: 0.794 Kaggle** | Exp 102 | 2 epochs, attention pooling critical |
| SED +0.112 cmAP over clip-level | murilo/Exp 016 | Framewise detection captures brief calls |
| Focal overfitting hurts LB | Exp 102 | More training epochs → lower Kaggle LB |
| ConvNeXt overfits focal data | Exp 102 | 0.644 val/cmap → 0.603 Kaggle LB |
| Every v3 ablation regressed vs v1 | murilo/Exp 016 | Gap is NOT in weights, loss, or mixup domain |
| Mel resolution gap: 1024 vs 4096 n_fft | Analysis | Largest untested difference vs competitors |
| LoRA equalizes pretraining domains | murilo/Exps 006-009 | 5× smaller ProtoCLR matches 90M models |
| Soft labels hurt as training targets | Exp 014 | −0.098 cmAP; don't replace hard labels |
| Activity scores predict calls (0.913 AUROC) | Exp 101/102 | Spectral contrast 0.888, LR with 4 indices 0.913 |
| Perch alone = 0.825 LB | Competition | Foundation model matches our best trained model |
| Metadata priors are free gains | Competition analysis | Site/hour tables, inference-only, +0.01-0.02 est. |
| 2025 1st: pseudo-labeling = +0.06 | Winning solutions | PowerTransform × 4 rounds on strong baseline |

## Murilo's Experiments

| ID | Name | Status | Result |
|----|------|--------|--------|
| 001 | BirdSet Benchmark | Complete | ConvNeXT 0.48, EfficientNet 0.38 on HSN |
| 002 | Supervised ViT Baseline | Complete | 0.164 cmAP on HSN |
| 003 | Model Training | Complete | ConvNeXT-Tiny pretrained best (cmAP=0.180) |
| 004 | Frame-Based Tokenization | Complete | 128×4 vertical tokens: cmAP=0.118 (+37%) |
| 005 | Pretrained Audio Fine-Tuning | Complete | EAT frozen probe: cmAP=0.301 |
| 006 | LoRA Fine-Tuning EAT | Complete | **Best HSN: cmAP=0.536** |
| 007 | ConvNeXt-Base LoRA | Complete | cmAP=0.527, ImageNet nearly matches AudioSet |
| 009 | ProtoCLR LoRA Fine-Tuning | Complete | 0.942 AUROC, 5× smaller matches EAT/ConvNeXt |
| 010 | Submission Pipeline | Complete | ONNX + Kaggle kernel, 3 architectures |
| 011 | LoRA Baseline 2026 | Complete | ProtoCLR fulldata 0.736 LB |
| 012 | Soundscape Fine-Tuning | Complete | ProtoCLR **0.767 LB**, all 3 models improved |
| 015 | Student Training (pseudo-labels) | **Rejected** | Pseudo-labels hurt with clip-level arch |
| 016 | SED Architecture | **Complete** | **0.827 LB** (v1-10s + pp2), v3 ablation done |
| 017 | Strong Baseline (mel overhaul) | **Proposed** | [Concept](concepts/strong-baseline-mel-overhaul.md) ready |

Full experiment history with detailed metrics: [EXPERIMENTS.md](../user/murilo/EXPERIMENTS.md)
All model runs and checkpoints: [MODEL-RUNS.md](../user/murilo/MODEL-RUNS.md)

## Anthony's Experiments

| ID | Name | Status | Result |
|----|------|--------|--------|
| 001 | Encoder Benchmark | Complete | BEATs 95.8%, EAT+LoRA 94.6% |
| 009 | BirdCLEF End-to-End Pipeline | In Progress | SSL pretrain done, finetune pending |
| 010 | Teacher Labels | Complete | BirdNET + Perch + BamBird (88 GB) |
| 011 | Prompted SED | Phase 0 Done | Inconclusive — emerging signal |
| 012 | EfficientNet-B1 Linear Probe | Complete | 0.649 Kaggle (frozen backbone) |
| 013 | Weak Supervision (FlyingSquid) | Complete | Soft labels generated |
| 014 | EfficientNet + Soft Labels | Rejected | Soft labels hurt (−0.098 cmAP) |
| 015 | EfficientNet SED + Soft Labels | Rejected | SED needs trainable backbone |
| 016 | Contrastive SED / Prototypical | Complete | Step 0 done |
| 100 | BirdCLEF 2026 EDA | Complete | 206 species, amphibian-dominated, 499x imbalance |
| 101 | Soundscape Label Calibration | Complete | Ensemble 0.880 AUROC, spectral contrast 0.888 |
| 102 | Multi-Backbone LoRA Ladder | **In Progress** | **0.794 Kaggle LB**, attention pooling, teacher filtering |

## Concept Docs

| Concept | Key Idea | Status |
|---------|----------|--------|
| [Strong Baseline Mel Overhaul](concepts/strong-baseline-mel-overhaul.md) | Close SED gap via mel resolution + recipe alignment | **Developing — exp 017 next** |
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

## Priority Stack

1. **Pseudo-labeling soundscapes** — run teacher on 10,592 unlabeled files, design labeling strategy
2. **Exp 017: Mel spectrogram overhaul** — n_fft=4096, 224 mels, single-variable test from v1
3. **Recipe alignment** — normalization, AttBlockV2 head, StochasticDepth, backbone scaling
4. **Free inference wins** — model soup, metadata priors (site/hour), temperature scaling
5. **Ensemble** — SED + LoRA attention + Perch hybrid

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
