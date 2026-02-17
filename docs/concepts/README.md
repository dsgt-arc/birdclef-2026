# Concepts

Research ideas, hypotheses, and directions being explored for BirdCLEF 2026.

## Concept Map

```
                         ┌──────────────────────────┐
                         │     Research Thesis       │
                         │  Foundation model quality │
                         │  under CPU inference      │
                         │  constraints (via Mamba)  │
                         └────────────┬─────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
  ┌──────────────┐          ┌──────────────────┐         ┌───────────────┐
  │  Architecture │          │  Training Data   │         │ Domain Gap    │
  │  & Backbone   │          │  & Labels        │         │ & Inference   │
  └──────┬───────┘          └────────┬─────────┘         └──────┬────────┘
         │                           │                          │
    ssamba-audio-mamba          weak-supervision-          soundscape-
         │                    label-fusion                domain-shift
    pretrained-mamba-              │                          │
    transfer                  unsupervised-event-       prompted-sound-
         │                    detection-labeling         event-detection ←──┐
    cnn-mamba-hybrid-                                        │              │
    frontend                                            (exp 011)          │
                                                             │              │
                                                        episode mixup ──────┘
                                                        + metric learning
```

## Index

### Active: SSAMBA / Mamba Pipeline

| Concept | Status | Description |
|---------|--------|-------------|
| [ssamba-audio-mamba](ssamba-audio-mamba.md) | **mature** | SSAMBA architecture overview, bidirectionality, experimental results (exps 004-009) |
| [pretrained-mamba-transfer](pretrained-mamba-transfer.md) | developing | Pre-trained weight transfer with re-initialized tokenizer; bidirectional conditioning analysis |
| [cnn-mamba-hybrid-frontend](cnn-mamba-hybrid-frontend.md) | nascent | LEAF + frequency attention tokenizer to replace linear patch embed |
| [prompted-sound-event-detection](prompted-sound-event-detection.md) | **developing** | Reference clip as Mamba prompt for few-shot/zero-shot SED; episode mixup; metric learning framing |

### Active: Labels & Supervision

| Concept | Status | Description |
|---------|--------|-------------|
| [weak-supervision-label-fusion](weak-supervision-label-fusion.md) | nascent | FlyingSquid triplet method to fuse BirdNET/Perch/energy into calibrated soft labels |
| [unsupervised-event-detection-labeling](unsupervised-event-detection-labeling.md) | developing | bambird/scikit-maad for per-chunk temporal event labels |

### Active: Domain & Context

| Concept | Status | Description |
|---------|--------|-------------|
| [soundscape-domain-shift](soundscape-domain-shift.md) | developing | Focal→soundscape gap: MixIT, ProtoCLR, augmentation, TTA strategies |
| [context-recovery](context-recovery.md) | developing | Decoupled representation + sequence model for long-form audio |
| [temporal-resolution-sequence-length](temporal-resolution-sequence-length.md) | nascent | Token budget vs frame resolution trade-offs for transformers/Mamba/CNNs |

### Background: Earlier Explorations (Pre-Mamba)

| Concept | Status | Description |
|---------|--------|-------------|
| [research-leads](research-leads.md) | archived | Curated repos/papers for discrete audio token pipeline (pre-Mamba era) |
| [data-and-compute-scale](data-and-compute-scale.md) | archived | Dataset sizes, model params, Kaggle constraints across BirdCLEF years |
| [ssast-audio-transformer](ssast-audio-transformer.md) | archived | SSAST evaluation — superseded by SSAMBA |
| [mlp-mixer-distillation](mlp-mixer-distillation.md) | archived | ViT→MLP-Mixer distillation — superseded by Mamba approach |
| [distilled-sequence-model](distilled-sequence-model.md) | archived | Distill BEATs→lightweight sequence model — superseded by SSAMBA direct use |
| [lora-finetuning-audio](lora-finetuning-audio.md) | archived | LoRA for audio transformers — explored in exp 001 |
| [discrete-audio-token-sequence-model](discrete-audio-token-sequence-model.md) | archived | Neural audio codec tokens for bioacoustics — abandoned |
| [supervised-vit-audio-pipeline](supervised-vit-audio-pipeline.md) | archived | Supervised ViT on BirdSet — completed in murilo/exp 002 |
| [vit-mae-bioacoustics-pretraining](vit-mae-bioacoustics-pretraining.md) | deferred | ViT+MAE self-supervised pretraining — deferred in favor of SSAMBA SSL |
| [pretraining-compute-feasibility](pretraining-compute-feasibility.md) | archived | PACE GPU budget analysis for pretraining strategies |

## Key Relationships

- **ssamba-audio-mamba** → **pretrained-mamba-transfer** → **prompted-sound-event-detection**: architecture → transfer mechanics → application
- **prompted-sound-event-detection** depends on **weak-supervision-label-fusion** for training labels (BirdNET/Perch teacher pseudo-labels)
- **soundscape-domain-shift** is the core problem; **prompted-sound-event-detection** is the proposed solution (reference conditioning bridges the gap)
- **cnn-mamba-hybrid-frontend** is orthogonal — can be combined with any downstream approach

## Status Definitions

- **nascent**: Initial idea, minimal exploration
- **developing**: Active exploration, hypotheses forming
- **mature**: Validated through experiments, clear findings
- **archived**: Explored and concluded or superseded
- **deferred**: Planned but postponed
