# Project Status

> **Last Updated**: February 2026

## Research Thesis

Foundation models like BEATs/AudioMAE learn superior bioacoustic representations but are excluded from BirdCLEF due to CPU inference constraints. This project explores strategies to unlock foundation model quality under CPU limits — currently via SSAMBA (State Space Audio Mamba), a Mamba-based architecture that processes audio in linear time.

## Current State

**Phase**: SSAMBA Backbone Validation — Pre-training Confirmed

The project has pivoted from transformer distillation (exps 001-003) to Mamba-based audio models (exps 004-007). Experiment 007 confirmed that **AudioSet pre-training is the dominant factor**: pre-trained SSAMBA Tiny achieves 0.18 cmAP, doubling the from-scratch baseline (0.086). However, a large gap remains to ConvNeXt (0.48) and SOTA (0.55).

### Honest Assessment

Pre-training closed the gap from 0.086 to 0.18 cmAP, confirming the architecture can learn when given good initialization. The remaining gap to competitive models (~0.47-0.55) likely comes from:

- **Model capacity**: SSAMBA Tiny (7M params) vs ConvNeXt/ViT-Base (85M+). The model memorizes training data (train cmAP >0.95) but can't generalize.
- **Pre-training domain**: AudioSet is generic audio; bird-specific pre-training (Bird-MAE) achieves 0.55 cmAP.
- **Architecture efficiency**: Mamba's selective scan is slower per-parameter than attention/convolution on GPU, limiting the model size we can practically train.

**Next priority**: determine whether scaling up (larger SSAMBA) or hybridizing (CNN frontend + Mamba backbone) can close the gap, or whether a different architecture is needed.

## Key Findings

| Finding | Source | Implication |
|---------|--------|-------------|
| BEATs 95.8% on ESC-50 (linear probe) | Exp 001 | Best frozen feature extractor |
| Static embeddings lose 48% accuracy | Exp 002 | Model2vec approach fails for transformers |
| ConvNeXT achieves 0.48 cmAP on HSN | murilo/Exp 001 | Strong baseline from pre-trained convolutions |
| ViT from scratch: 0.164 cmAP on HSN | murilo/Exp 002 | Pre-training matters more than architecture |
| ConvNeXT-Tiny pretrained cmAP=0.180 on HSN | [murilo/Exp 003](../user/murilo/experiments/003-model-training/results.md) | Best supervised baseline, beats ViT-Base (85M) with 3x fewer params |
| ImageNet pretraining gives 3.0x lift (ConvNeXT) | [murilo/Exp 003](../user/murilo/experiments/003-model-training/results.md) | Pretraining critical for small datasets |
| CNNs transfer better than ViTs to spectrograms | [murilo/Exp 003](../user/murilo/experiments/003-model-training/results.md) | Hierarchical conv features > patch tokenization for audio |
| SSAMBA 30s windows: **0.08 cmAP** on HSN | Exp 006 | Far below baselines — backbone needs diagnosis |
| SSAMBA pre-trained: **0.18 cmAP** on HSN | Exp 007 | Pre-training doubles performance; Mamba-only transfer ≈ full transfer |
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
| anthony/006 | SSAMBA BirdSet HSN | Complete | 0.086 cmAP from scratch — far below baselines |
| anthony/007 | Pre-trained SSAMBA Transfer | **Complete** | 0.179 cmAP (16x16), 0.172 cmAP (128x1) — pre-training confirmed |
| anthony/008 | Attention-Pooled SED | **Proposed** | SED attention pooling + frequency attention tokenizer + nnAudio |
| murilo/001 | BirdSet Benchmark | Complete | ConvNeXT 0.48, EfficientNet 0.38 on HSN |
| murilo/002 | Supervised ViT Baseline | Complete | 0.164 cmAP on HSN |
| murilo/003 | [Model Training](../user/murilo/experiments/003-model-training/) | Complete | ConvNeXT-Tiny pretrained best (cmAP=0.180), 3x pretraining lift, CNNs > ViTs for spectrograms |

## Concept Docs

Research ideas explored but not yet validated:

| Concept | Key Idea | Dependency |
|---------|----------|------------|
| [Prompted SED](concepts/prompted-sound-event-detection.md) | Reference audio as Mamba prefix for few-shot detection | Needs working SSAMBA backbone |
| [Weak Supervision Label Fusion](concepts/weak-supervision-label-fusion.md) | FlyingSquid triplet method to fuse noisy teacher signals | Needs ground-truth validation set |
| [Unsupervised Event Detection](concepts/unsupervised-event-detection-labeling.md) | bambird for per-token temporal labels | Available now via BirdSet |
| [Soundscape Domain Shift](concepts/soundscape-domain-shift.md) | XenoCanto → soundscape gap analysis | Fundamental challenge |
| [Context Recovery](concepts/context-recovery.md) | Using temporal context for predictions | Addressed by 30s windows |

**Key insight from exp 007**: Pre-training is confirmed as the dominant factor. The backbone can learn — the question is now about capacity and generalization, not fundamental viability.

## Priority Stack

1. **Close the capacity gap** — SSAMBA Tiny (7M params) hits 0.18 cmAP with severe overfitting. Investigate whether larger SSAMBA variants or hybrid CNN+Mamba architectures can improve generalization.
2. **Explore CNN frontend** — Replace raw mel patches with a learned CNN tokenizer (see concepts/cnn-mamba-hybrid-frontend.md) to give Mamba better input features.
3. **Pursue downstream tasks** — With a validated backbone, revisit token-level SED and multi-signal approaches on a foundation that can learn.

## Quick Links

| Resource | Link |
|----------|------|
| Concepts | [concepts/](concepts/) |
| Literature | [references/literature.md](references/literature.md) |
| Vendor Docs | [vendor/](vendor/) |
| Templates | [_templates/](_templates/) |
