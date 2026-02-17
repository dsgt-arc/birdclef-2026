# Project Status

> **Last Updated**: 2026-02-17

## Research Thesis

Foundation models like BEATs/AudioMAE learn superior bioacoustic representations but are excluded from BirdCLEF due to CPU inference constraints. This project explores strategies to unlock foundation model quality under CPU limits — currently via SSAMBA (State Space Audio Mamba), a Mamba-based architecture that processes audio in linear time.

## Current State

**Phase**: SSL Pre-training + Teacher Label Pipeline

The project has validated SSAMBA as a viable backbone (exp 007: 0.18 cmAP with AudioSet pre-training, 2x over scratch). Attention-pooled SED did not improve over mean pooling (exp 008). Current focus is on:

1. **SSL pre-training** (exp 009) — masked patch generation (MPG) with L1 loss + linear decoder, training from scratch and with AudioSet warm-start
2. **Teacher labels** (exp 010) — BamBird, BirdNET, and Perch labels generated for all BirdCLEF 2025 data (88 GB of predictions/embeddings)
3. **Prompted SED** (exp 011) — proposed next experiment using reference clips as Mamba prompts for per-token detection

### Honest Assessment

Pre-training closed the gap from 0.086 to 0.18 cmAP, but a large gap remains to ConvNeXt (0.48) and SOTA (0.55). The bottleneck appears to be representation quality — not pooling strategy (exp 008) or model size. SSL pre-training on BirdCLEF data (exp 009) and teacher-guided training (exp 010→011) are the current strategies to close this gap.

## Key Findings

| Finding | Source | Implication |
|---------|--------|-------------|
| BEATs 95.8% on ESC-50 (linear probe) | Exp 001 | Best frozen feature extractor |
| Static embeddings lose 48% accuracy | Exp 002 | Model2vec approach fails for transformers |
| ConvNeXT achieves 0.48 cmAP on HSN | murilo/Exp 001 | Strong baseline from pre-trained convolutions |
| SSAMBA from scratch: 0.086 cmAP on HSN | Exp 006 | Far below baselines — backbone needs pre-training |
| SSAMBA pre-trained: **0.18 cmAP** on HSN | Exp 007 | Pre-training doubles performance; Mamba-only transfer works |
| SED attention pooling: no improvement | Exp 008 | Bottleneck is representation quality, not pooling |
| On-the-fly nnAudio: no throughput penalty | Exp 008 | Can skip precomputed mel cache |
| BamBird clustering: 12.4% signal ratio | Exp 010 | Unreliable signal/noise labels, but good temporal boundaries |
| Teacher labels: 88 GB across 3 models | Exp 010 | Ready for weak supervision fusion |
| Soundscapes mostly quiet: 5-11% active | Exp 010 | Need careful episode sampling for soundscape training |

## Experiments

| ID | Name | Status | Result |
|----|------|--------|--------|
| anthony/001 | Encoder Benchmark | Complete | BEATs 95.8%, EAT+LoRA 94.6% |
| anthony/002 | BEATs Inference Optimization | Complete | Rejected — 63% best accuracy |
| anthony/003 | Model Distillation | Proposed | Distill BEATs → lightweight student |
| anthony/004 | SSAMBA HF Port | Complete | SSAMBA ported to HuggingFace format |
| anthony/005 | Token Context Shaping | Complete | Rectangular patches + ONNX benchmark |
| anthony/006 | SSAMBA BirdSet HSN | Complete | 0.086 cmAP from scratch |
| anthony/007 | Pre-trained SSAMBA Transfer | Complete | 0.179 cmAP (16x16), 0.173 cmAP (128x1) |
| anthony/008 | Attention-Pooled SED | **Rejected** | No improvement over mean pooling (0.164 vs 0.173) |
| anthony/009 | BirdCLEF End-to-End Pipeline | **In Progress** | SSL pretrain (MPG) + finetune pipeline |
| anthony/010 | Teacher Labels | **Complete** | BamBird + BirdNET + Perch on full dataset (88 GB) |
| anthony/011 | Prompted SED | **Proposed** | Reference clip as Mamba prompt for per-token SED |
| murilo/001 | BirdSet Benchmark | Complete | ConvNeXT 0.48, EfficientNet 0.38 on HSN |
| murilo/002 | Supervised ViT Baseline | Complete | 0.164 cmAP on HSN |
| murilo/003 | Model Training | Complete | ConvNeXT-Tiny pretrained best (cmAP=0.180) |

## Concept Docs

Research ideas explored but not yet validated:

| Concept | Key Idea | Dependency |
|---------|----------|------------|
| [Prompted SED](concepts/prompted-sound-event-detection.md) | Reference audio as Mamba prefix for few-shot detection | Exp 009 checkpoint + exp 010 labels |
| [Weak Supervision Label Fusion](concepts/weak-supervision-label-fusion.md) | FlyingSquid triplet method to fuse noisy teacher signals | Exp 010 labels (ready) |
| [Unsupervised Event Detection](concepts/unsupervised-event-detection-labeling.md) | bambird for per-token temporal labels | Exp 010 (done — 660k ROIs) |
| [Soundscape Domain Shift](concepts/soundscape-domain-shift.md) | XenoCanto → soundscape gap analysis | Fundamental challenge |
| [Context Recovery](concepts/context-recovery.md) | Using temporal context for predictions | Addressed by 30s windows |

## Priority Stack

1. **Complete SSL pre-training** (exp 009) — pretrain-full and pretrain-warm runs in progress
2. **Finetune with SSL backbone** (exp 009) — compare SSL-pretrained vs AudioSet-pretrained on BirdSet HSN
3. **Prompted SED** (exp 011) — reference clip conditioning for per-token detection, using exp 010 teacher labels
4. **Label fusion** — FlyingSquid with BamBird + BirdNET + Perch as 3 weak labeling functions
5. **Kaggle submission** — end-to-end pipeline tested (kernel v7 submitted)

## Quick Links

| Resource | Link |
|----------|------|
| Concepts | [concepts/](concepts/) |
| Literature | [references/literature.md](references/literature.md) |
| Vendor Docs | [vendor/](vendor/) |
| Templates | [_templates/](_templates/) |
