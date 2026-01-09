# Project Status

> **Last Updated**: January 2026

## Research Thesis

Foundation models like BEATs/AudioMAE learn superior bioacoustic representations but are excluded from BirdCLEF due to CPU inference constraints. This project explores strategies to unlock foundation model quality under CPU limits.

## Current State

**Phase**: Research & Literature Review

We are currently evaluating foundation models and distillation strategies. No experiments have been run yet.

## Key Findings So Far

| Finding | Source | Implication |
|---------|--------|-------------|
| EAT is 15x more efficient than BEATs | [literature.md](references/literature.md) | May be better distillation target |
| OpenBEATs (300M) is SOTA on bioacoustics | [literature.md](references/literature.md) | Best available teacher model |
| BirdMAE wins on BirdSet benchmark | [literature.md](references/literature.md) | Domain-specific training helps |
| Focal→soundscape domain gap is critical | [soundscape-domain-shift.md](concepts/soundscape-domain-shift.md) | Must address in pipeline |

## Active Research Directions

1. **Distillation target selection** - EAT vs OpenBEATs vs BirdMAE
2. **CPU-viable architectures** - MLP-Mixer, model2vec, quantization
3. **Domain adaptation** - ProtoCLR, MixIT denoising

See [concepts/research-leads.md](concepts/research-leads.md) for full list.

## What Works (Baselines)

| System | Performance | CPU Viable? |
|--------|-------------|-------------|
| BirdNET | Production-grade, 6500 species | Yes |
| Perch | 15k species, EfficientNet-B3 | Yes |
| BEATs | SOTA representations | No (too slow) |

## Experiments

Experiments are tracked per-user in `user/<name>/experiments/`. See individual user workspaces.

## Next Steps

1. Benchmark foundation models (BEATs, EAT, wav2vec2) on AudioSet bird classes
2. Deep dive on EAT architecture
3. Prototype distillation pipeline based on best encoder

## Quick Links

| Resource | Link |
|----------|------|
| Concepts | [concepts/](concepts/) |
| Literature | [references/literature.md](references/literature.md) |
| Vendor Docs | [vendor/](vendor/) |
| Templates | [_templates/](_templates/) |
