# Concepts

Research ideas, hypotheses, and directions being explored for BirdCLEF 2026.

## Concept Map

```
                    ┌─────────────────────┐
                    │   Research Thesis   │
                    │  Distill foundation │
                    │  models to CPU      │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │   Domain    │    │  Efficient  │    │   Model     │
    │ Adaptation  │    │Architectures│    │ Selection   │
    └─────────────┘    └─────────────┘    └─────────────┘
           │                   │                   │
           ▼                   ▼                   ▼
    soundscape-       mlp-mixer-         research-leads
    domain-shift      distillation       (EAT, OpenBEATs,
                                          BirdMAE)
```

## Index

| Concept | Status | Description |
|---------|--------|-------------|
| [soundscape-domain-shift](soundscape-domain-shift.md) | developing | Focal→soundscape gap and mitigation strategies |
| [data-and-compute-scale](data-and-compute-scale.md) | developing | Dataset sizes, model params, Kaggle constraints |
| [mlp-mixer-distillation](mlp-mixer-distillation.md) | nascent | ViT→MLP-Mixer distillation for CPU inference |
| [research-leads](research-leads.md) | developing | Active model candidates and next steps |
| [vit-mae-bioacoustics-pretraining](vit-mae-bioacoustics-pretraining.md) | **mature** | Self-supervised ViT+MAE pre-training for BirdSet (Experiments 002-003) |

## Status Definitions

- **nascent**: Initial idea, minimal exploration
- **developing**: Active exploration, hypotheses forming
- **mature**: Ready for experiments, clear hypotheses
- **archived**: Explored and concluded

## Adding a New Concept

Use the template at [`../_templates/concept.md`](../_templates/concept.md).
