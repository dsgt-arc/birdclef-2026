# Project Status

> **Last Updated**: 2026-03-14

## Research Thesis

BirdCLEF 2025 requires CPU-only inference on soundscapes. This project explores two parallel tracks:
1. **EfficientNet-B1** — frozen pre-trained backbone with trainable heads, currently the best Kaggle performer (0.649 private LB)
2. **SSAMBA** — Mamba-based linear-time backbone for potential future gains via SSL pre-training and prototypical classification

## Current State

**Phase**: EfficientNet refinement + Prototypical/Contrastive methods

The EfficientNet-B1 linear probe (exp 012) achieved **0.649 Kaggle private LB** with a frozen backbone — the project's best submission. Subsequent experiments showed that soft labels from weak supervision (exp 014) and SED heads (exp 015) did not improve over hard-label baselines. Current focus is on:

1. **Contrastive SED / prototypical classification** (exp 016) — testing if frozen EfficientNet features have metric structure for few-shot species ID
2. **Weak supervision pipeline** (exp 013) — completed FlyingSquid fusion of BirdNET + Perch + acoustic indices, available as soft labels but not yet beneficial
3. **SSAMBA SSL pre-training** (exp 009) — MPG pre-training runs completed, finetune evaluation pending

### Honest Assessment

EfficientNet-B1 frozen probe (0.649 Kaggle) is a strong baseline but well below top solutions (>0.85). The 2025 Kaggle landscape shows the dominant factor is **iterative pseudo-labeling** (3-4 rounds), not architecture. Soft labels from exp 013 hurt when used as training targets (exp 014/015), suggesting the bottleneck is label quality, not model capacity. The prototypical approach (exp 016) may unlock pseudo-labeling without needing soft labels as direct supervision.

## Key Findings

| Finding | Source | Implication |
|---------|--------|-------------|
| BEATs 95.8% on ESC-50 (linear probe) | Exp 001 | Best frozen feature extractor |
| Static embeddings lose 48% accuracy | Exp 002 | Model2vec approach fails for transformers |
| ConvNeXT achieves 0.48 cmAP on HSN | murilo/Exp 001 | Strong baseline from pre-trained convolutions |
| Vertical tokenization +37% over 16×16 | murilo/Exp 004 | 128×4 tokens: cmAP=0.118, frequency axis matters |
| EAT LoRA fine-tuning: **0.536 cmAP** | murilo/Exp 006 | 78% improvement over frozen probe; MLP head > LoRA scope |
| ConvNeXt LoRA: **0.527 cmAP** | murilo/Exp 007 | ImageNet features nearly match AudioSet (only 1.7% gap) |
| SSAMBA pre-trained: **0.18 cmAP** on HSN | Exp 007 | Pre-training doubles performance; Mamba-only transfer works |
| SED attention pooling: no improvement | Exp 008 | Bottleneck is representation quality, not pooling |
| Teacher labels: 88 GB across 3 models | Exp 010 | BirdNET + Perch + BamBird + acoustic indices on full dataset |
| Prompted SED phase 0: inconclusive | Exp 011 | Finetuned backbone has emerging species signal but below threshold |
| **EfficientNet-B1 frozen: 0.649 Kaggle** | Exp 012 | **Best submission** — frozen backbone + linear head |
| Weak supervision fusion complete | Exp 013 | FlyingSquid triplet method, 3-6 LFs per species |
| Soft labels hurt linear probe | Exp 014 | 0.238 cmAP vs 0.336 baseline — don't replace hard labels |
| SED + soft labels: no benefit | Exp 015 | 0.237 cmAP — bottleneck is label quality, not head architecture |
| Pseudo-labeling dominated 2025 solutions | Kaggle analysis | Top-1 used 3-4 rounds iterative pseudo-labeling |

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
| anthony/009 | BirdCLEF End-to-End Pipeline | **In Progress** | SSL pretrain (MPG) done, finetune pending |
| anthony/010 | Teacher Labels | **Complete** | BirdNET + Perch + BamBird + acoustic indices (88 GB) |
| anthony/011 | Prompted SED | **Phase 0 Done** | Inconclusive — emerging signal but below pass gate |
| anthony/012 | EfficientNet-B1 Linear Probe | **Complete** | **0.336 val/cmap, 0.649 Kaggle private** |
| anthony/013 | Weak Supervision (FlyingSquid) | **Complete** | Triplet fusion of 3-6 LFs, soft labels generated |
| anthony/014 | EfficientNet + Soft Labels | **Rejected** | 0.238 cmAP — soft labels hurt vs hard-label baseline |
| anthony/015 | EfficientNet SED + Soft Labels | **Rejected** | 0.237 cmAP — SED head adds nothing with soft labels |
| anthony/016 | Contrastive SED / Prototypical | **In Progress** | Step 0: zero-training baseline (awaiting results) |
| murilo/001 | BirdSet Benchmark | Complete | ConvNeXT 0.48, EfficientNet 0.38 on HSN |
| murilo/002 | Supervised ViT Baseline | Complete | 0.164 cmAP on HSN |
| murilo/003 | Model Training | Complete | ConvNeXT-Tiny pretrained best (cmAP=0.180) |
| murilo/004 | Frame-Based Tokenization | Complete | 128×4 vertical tokens: cmAP=0.118 (+37%) |
| murilo/005 | Pretrained Audio Fine-Tuning | Complete | EAT frozen probe: cmAP=0.301 |
| murilo/006 | LoRA Fine-Tuning EAT | **Complete** | **Best HSN: cmAP=0.536** |
| murilo/007 | ConvNeXt-Base LoRA | **Complete** | cmAP=0.527, ImageNet nearly matches AudioSet |
| murilo/008 | Frame Tokenization EAT | Proposed | Vertical patches for EAT |
| murilo/009 | ProtoCLR Fine-Tuning | Proposed | CvT-13 prototypical contrastive for ensemble diversity |
| adrian/001 | Inference Benchmark | Complete | CPU/GPU timing for BirdNET, Perch, BEATs, etc. |
| adrian/002 | Kaggle Submission | Complete | Offline submission pipeline (three artifacts approach) |

## Concept Docs

Research ideas explored but not yet validated:

| Concept | Key Idea | Status |
|---------|----------|--------|
| [Prompted SED](concepts/prompted-sound-event-detection.md) | Reference audio as Mamba prefix for few-shot detection | Tested (exp 011) — inconclusive |
| [Weak Supervision Label Fusion](concepts/weak-supervision-label-fusion.md) | FlyingSquid triplet method to fuse noisy teacher signals | Done (exp 013) — labels generated |
| [Unsupervised Event Detection](concepts/unsupervised-event-detection-labeling.md) | bambird for per-token temporal labels | Done (exp 010) — 660k ROIs |
| [Soundscape Domain Shift](concepts/soundscape-domain-shift.md) | XenoCanto → soundscape gap analysis | Fundamental challenge |
| [Context Recovery](concepts/context-recovery.md) | Using temporal context for predictions | Addressed by 30s windows |

## Priority Stack

1. **Evaluate exp 016 step 0** — does frozen EfficientNet have metric structure for prototypical classification?
2. **Prototypical/contrastive training** (exp 016 steps 1+) — if step 0 passes decision gates
3. **Iterative pseudo-labeling** — the dominant technique from 2025 top solutions, not yet attempted
4. **SSAMBA finetune with SSL backbone** (exp 009) — compare SSL-pretrained vs AudioSet-pretrained
5. **Kaggle submission refinement** — current best is 0.649 (exp 012), pipeline tested end-to-end

## Quick Links

| Resource | Link |
|----------|------|
| Concepts | [concepts/](concepts/) |
| Literature | [references/literature.md](references/literature.md) |
| Vendor Docs | [vendor/](vendor/) |
| Templates | [_templates/](_templates/) |
| Kaggle Notebooks | [references/kaggle/](references/kaggle/) |
