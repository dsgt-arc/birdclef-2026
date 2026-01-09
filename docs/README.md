# Documentation

Shared context and documentation for the BirdCLEF 2026 discrete token pipeline research project.

> **Research Thesis**: Foundation models like BEATs/AudioMAE learn superior bioacoustic representations but are excluded from BirdCLEF due to CPU inference constraints. This project explores strategies to unlock foundation model quality under CPU limits.

---

## Quick Start

**New here?** Start with:
1. [STATUS.md](STATUS.md) - What works now, what we're trying
2. [concepts/](concepts/) - Research ideas we're exploring
3. [references/literature.md](references/literature.md) - Key papers

---

## Directory Structure

```
docs/
├── STATUS.md           # Current state summary (start here)
├── concepts/           # Research ideas and hypotheses
│   ├── README.md       # Concept map and index
│   └── *.md            # Individual concepts
├── experiments/        # Experiment proposals and results
│   └── NNN-name/       # Each experiment gets a directory
├── references/
│   ├── literature.md   # Key papers with links
│   └── deep-research/  # AI research outputs and syntheses
├── vendor/             # External tool documentation
│   └── README.md       # Index of vendored repos
└── _templates/         # Document templates
    ├── concept.md
    ├── experiment-proposal.md
    ├── experiment-result.md
    └── reference.md
```

---

## Concepts (Research Ideas)

| Concept | Description |
|---------|-------------|
| [soundscape-domain-shift](concepts/soundscape-domain-shift.md) | Focal→soundscape domain gap: MixIT, ProtoCLR, augmentation |
| [data-and-compute-scale](concepts/data-and-compute-scale.md) | Dataset sizes, model params, Kaggle inference constraints |
| [mlp-mixer-distillation](concepts/mlp-mixer-distillation.md) | ViT→MLP-Mixer distillation for CPU efficiency |
| [research-leads](concepts/research-leads.md) | Active model candidates: EAT, OpenBEATs, BirdMAE |

---

## References

### Literature
- [references/literature.md](references/literature.md) - Key papers with arXiv links, prioritized reading list

### Deep Research (AI Syntheses)
| Doc | Description |
|-----|-------------|
| [wav2vec2-landscape](references/deep-research/wav2vec2-landscape.md) | Post-wav2vec 2.0 era: codecs, transformers, Mamba |
| [birdclef-history](references/deep-research/birdclef-history.md) | BirdCLEF competition history 2014-2025 |
| [beats-architecture](references/deep-research/beats-architecture.md) | Deep dive on BEATs iterative tokenization |

### Vendor Documentation
See [vendor/README.md](vendor/README.md) for 15+ vendored repositories with:
- Pipeline stage mapping (tokenization, representation learning, embedding, inference)
- Category and status for each repository

---

## Current Focus (Jan 2026)

Based on recent literature review:

1. **EAT** - 15x more efficient than BEATs, similar performance
2. **OpenBEATs** - SOTA on bioacoustics, fully open-source (300M params)
3. **BirdMAE** - Best on BirdSet benchmark, bird-specific training

Key question: Can these foundation models be distilled to CPU-viable inference while retaining bioacoustics performance?

---

## Quick Links

| Resource | Link |
|----------|------|
| Project Status | [STATUS.md](STATUS.md) |
| Concept Index | [concepts/README.md](concepts/README.md) |
| Key Papers | [references/literature.md](references/literature.md) |
| Vendor Index | [vendor/README.md](vendor/README.md) |
| Templates | [_templates/](_templates/) |
