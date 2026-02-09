# Project Status

> **Last Updated**: February 2026

## Research Thesis

Foundation models like BEATs/AudioMAE learn superior bioacoustic representations but are excluded from BirdCLEF due to CPU inference constraints. This project explores strategies to unlock foundation model quality under CPU limits — currently via SSAMBA (State Space Audio Mamba), a Mamba-based architecture that processes audio in linear time.

## Current State

**Phase**: SSAMBA Backbone Validation

The project has pivoted from transformer distillation (exps 001-003) to Mamba-based audio models (exps 004-006). The SSAMBA architecture has been ported to HuggingFace format with rectangular patch support, but **training results are poor** — 0.08 cmAP on BirdSet HSN vs. ConvNeXT's 0.48. Before pursuing more sophisticated approaches (SED, multi-signal, prompted detection), we need to diagnose why the backbone underperforms.

### Honest Assessment

The gap between SSAMBA (0.08) and ConvNeXT (0.48) is too large to attribute to any single factor. Possible causes include:

- **No pre-training**: SSAMBA is trained from scratch; ConvNeXT uses ImageNet pre-training. This is likely the dominant factor.
- **Architecture suitability**: Mamba may not be the right inductive bias for audio classification. The sequential scan assumption may not capture the 2D structure of spectrograms as well as convolutions or attention.
- **Training dynamics**: SSAMBA Tiny (7.4M params, 192-dim, 24 layers) may have poor optimization landscape for this task. Learning rate, warmup, and batch size haven't been systematically tuned.
- **Data pipeline issues**: Exp 006 fixed the no-negatives problem (30s windows with padding), but there may be other data-side issues.

**The highest-priority work is diagnostic**: understand *why* SSAMBA fails before building more on top of it. Concepts like prompted SED and weak supervision are theoretically motivated but premature without a functional backbone.

## Key Findings

| Finding | Source | Implication |
|---------|--------|-------------|
| BEATs 95.8% on ESC-50 (linear probe) | Exp 001 | Best frozen feature extractor |
| Static embeddings lose 48% accuracy | Exp 002 | Model2vec approach fails for transformers |
| ConvNeXT achieves 0.48 cmAP on HSN | Murilo exp 001 | Strong baseline from pre-trained convolutions |
| ViT from scratch: 0.164 cmAP on HSN | Murilo exp 002 | Pre-training matters more than architecture |
| SSAMBA 30s windows: **0.08 cmAP** on HSN | Exp 006 | Far below baselines — backbone needs diagnosis |
| SSAMBA forward pass ~28ms up to 3000 tokens | Exp 005 | Speed is not the bottleneck; learning is |
| Pure PyTorch SSM too slow for >1000 tokens | Exp 004 | CUDA kernels (mamba-ssm) required |

## Experiments

| ID | Name | Status | Result |
|----|------|--------|--------|
| anthony/001 | Encoder Benchmark | Complete | BEATs 95.8%, EAT+LoRA 94.6% |
| anthony/002 | BEATs Inference Optimization | Complete | Rejected — 63% best accuracy |
| anthony/003 | Model Distillation | Proposed | Distill BEATs → lightweight student |
| anthony/004 | SSAMBA HF Port | Complete | SSAMBA ported to HuggingFace format |
| anthony/005 | Token Context Shaping | Complete | Rectangular patches + ONNX benchmark |
| anthony/006 | SSAMBA BirdSet HSN | **In Progress** | 0.08 cmAP — far below baselines |
| anthony/007 | Token-Level SED | Proposed | Per-token supervision via detected_events |
| anthony/008 | Multi-Signal SED | Proposed | ConvNeXT distillation + bambird + SSL |
| murilo/001 | BirdSet Benchmark | Complete | ConvNeXT 0.48, EfficientNet 0.38 on HSN |
| murilo/002 | Supervised ViT Baseline | Complete | 0.164 cmAP on HSN |
| murilo/003 | Audio-MAE Pre-training | Deferred | Planning complete, archived |

## Concept Docs

Research ideas explored but not yet validated:

| Concept | Key Idea | Dependency |
|---------|----------|------------|
| [Prompted SED](concepts/prompted-sound-event-detection.md) | Reference audio as Mamba prefix for few-shot detection | Needs working SSAMBA backbone |
| [Weak Supervision Label Fusion](concepts/weak-supervision-label-fusion.md) | FlyingSquid triplet method to fuse noisy teacher signals | Needs ground-truth validation set |
| [Unsupervised Event Detection](concepts/unsupervised-event-detection-labeling.md) | bambird for per-token temporal labels | Available now via BirdSet |
| [Soundscape Domain Shift](concepts/soundscape-domain-shift.md) | XenoCanto → soundscape gap analysis | Fundamental challenge |
| [Context Recovery](concepts/context-recovery.md) | Using temporal context for predictions | Addressed by 30s windows |

**Critical gap**: All concept docs assume a working backbone. None address the question of whether SSAMBA is the right architecture.

## Priority Stack

1. **Diagnose SSAMBA failure** — Is it pre-training, architecture, optimization, or data? Run controlled comparisons:
   - SSAMBA with pre-trained weights (if available) vs. from scratch
   - Learning rate / warmup sweep on current setup
   - Compare SSAMBA vs. a simple ConvNet of similar size trained from scratch on the same data
2. **Decide go/no-go on SSAMBA** — If a similarly-sized ConvNet from scratch beats SSAMBA, the architecture may not be suitable and the Mamba bet needs reassessment
3. **Only then**: pursue exps 007/008 and the concept ideas, on a backbone that's shown it can learn

## Quick Links

| Resource | Link |
|----------|------|
| Concepts | [concepts/](concepts/) |
| Literature | [references/literature.md](references/literature.md) |
| Vendor Docs | [vendor/](vendor/) |
| Templates | [_templates/](_templates/) |
