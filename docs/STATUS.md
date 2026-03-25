# Project Status

> **Last Updated**: 2026-03-25

## Current State

**Best LB**: **0.826** (EfficientNet-B0 SED v1 + post-processing + 10s inference, exp 016)
**Target**: 0.85+ labeled-data baseline, then 0.90+ with pseudo-labeling
**Active branch**: `mg/016-sed-improvements`
**Next experiment**: Exp 017 — mel spectrogram overhaul ([concept](concepts/strong-baseline-mel-overhaul.md))

### Score Progression

| Model | Experiment | LB | Notes |
|-------|-----------|-----|-------|
| ProtoCLR clip-level | 012 | 0.767 | Best clip-level model |
| EfficientNet-B0 SED v1 | 016 | 0.795 | SED architecture validated |
| + post-processing (TTA + smoothing + TopN) | 016 | 0.823 | +0.028 free gains |
| **+ 10s overlapping inference** | **016** | **0.826** | **Current best** |
| Competitor SED (single model) | — | 0.908 | Our target |
| Perch + ProtoSSM | — | 0.913 | Foundation model approach |
| Pantanal Distill | — | 0.917 | Best public notebook |

### What's Next

**Exp 017: Strong Baseline via Mel Overhaul** — close the 0.082 gap to 0.908 using only labeled data.

The v3 ablation (exp 016) proved no single recipe tweak helps — the gap is likely cumulative,
starting with **mel spectrogram resolution** (n_fft=1024/128 mels vs 2025 1st place n_fft=4096/224 mels).
Phased plan: mel overhaul → normalization → AttBlockV2 head → backbone scaling → ASL loss.
See [concept doc](concepts/strong-baseline-mel-overhaul.md) for full details.

After establishing a strong labeled-data baseline, **Phase 2** will leverage the 10,592 unlabeled
soundscapes via iterative pseudo-labeling with PowerTransform (2025 1st place: +0.06 from this alone).

### Honest Assessment

Our SED architecture (GeM + AttHead) matches 3/4 BirdCLEF winners. The v3 ablation
exhaustively tested weight variants, loss functions, and mixup domains — all regressed.
The remaining gap is likely in input resolution (mel params), training recipe details
(regularization, head design), and data leverage (pseudo-labeling, metadata priors).
The Perch-based approaches (0.907-0.917) represent a fundamentally different paradigm
(frozen foundation model + lightweight heads) that is complementary to our trained SED approach.

## Key Findings

| Finding | Source | Implication |
|---------|--------|-------------|
| **SED v1 = 0.826 LB** (best single model) | murilo/Exp 016 | SED + post-processing + 10s inference |
| SED +0.112 cmAP over clip-level | murilo/Exp 016 | Framewise detection captures brief calls |
| Every v3 ablation regressed vs v1 | murilo/Exp 016 | Gap is NOT in weights, loss, or mixup domain |
| Mel resolution gap: 1024 vs 4096 n_fft | Analysis | Largest untested difference vs competitors |
| ProtoCLR 0.767 (best clip-level) | murilo/Exp 012 | Soundscape fine-tuning universally helps |
| LoRA equalizes pretraining domains | murilo/Exps 006-009 | 5× smaller ProtoCLR matches 90M models |
| Soft labels hurt as training targets | Exp 014 | −0.098 cmAP; don't replace hard labels |
| Pseudo-labels (round 0) didn't help | murilo/Exp 015 | Clip-level architecture was the bottleneck |
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
| 016 | SED Architecture | **Complete** | **0.826 LB** (v1-10s + pp), v3 ablation done |
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
| 016 | Contrastive SED / Prototypical | In Progress | Step 0 awaiting results |

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

1. **Exp 017: Mel spectrogram overhaul** — n_fft=4096, 224 mels, single-variable test from v1
2. **Exp 017: Recipe alignment** — normalization, AttBlockV2 head, StochasticDepth, backbone scaling
3. **Free inference wins** — model soup, metadata priors (site/hour), temperature scaling
4. **Pseudo-labeling** (Phase 2) — PowerTransform × 3-4 rounds on the strongest baseline
5. **Ensemble** — SED + Perch hybrid, OpenVINO for CPU budget

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
