# Documentation

Shared context and documentation for the BirdCLEF 2026 discrete token pipeline research project.

> **Research Thesis**: Foundation models like BEATs/AudioMAE learn superior bioacoustic representations but are excluded from BirdCLEF due to CPU inference constraints. This project explores strategies to unlock foundation model quality under CPU limits.

---

## Table of Contents

### Research (start here)

| Doc | Description |
|-----|-------------|
| [00_references.md](research/00_references.md) | **Key papers with arXiv links** - prioritized reading list |
| [01_initial_research.md](research/01_initial_research.md) | Post-wav2vec 2.0 landscape: codecs, transformers, Mamba |
| [02_leads.md](research/02_leads.md) | **Research leads** - models to explore, updated Dec 2025 |
| [03_soundscape_domain_shift.md](research/03_soundscape_domain_shift.md) | Focal→soundscape domain gap: MixIT, ProtoCLR, augmentation |
| [04_data_and_compute_scale.md](research/04_data_and_compute_scale.md) | Dataset sizes, model params, Kaggle inference constraints |
| [05_birdclef.md](research/05_birdclef.md) | BirdCLEF competition history 2014-2025, winning techniques |
| [06_beats.md](research/06_beats.md) | **Deep dive: BEATs** - iterative tokenization, architecture |

### Architecture

| Doc | Description |
|-----|-------------|
| [mlp-mixer-distillation.md](architecture/mlp-mixer-distillation.md) | Alternative path: ViT→MLP-Mixer distillation for efficiency |

### Vendor Documentation

See [vendor/README.md](vendor/README.md) for all 15 vendored repositories with:
- Pipeline stage mapping (tokenization, representation learning, embedding, inference)
- Category and status for each repository

---

## Directory Structure

```
docs/
├── architecture/     # System design, model architectures
├── experiments/      # Experiment results, ablation studies
├── research/         # Literature reviews, research notes
└── vendor/           # Documentation for vendored repositories
```

## Quick Links

| Resource | Link |
|----------|------|
| Key References | [research/00_references.md](research/00_references.md) |
| Research Leads | [research/02_leads.md](research/02_leads.md) |
| Vendor Index | [vendor/README.md](vendor/README.md) |
| Pipeline Proposal | [openspec/changes/add-discrete-token-pipeline/](../openspec/changes/add-discrete-token-pipeline/proposal.md) |

## Current Focus (Dec 2025)

Based on recent literature review:

1. **EAT** - 15× more efficient than BEATs, similar performance
2. **OpenBEATs** - SOTA on bioacoustics, fully open-source (300M params)
3. **BirdMAE** - Best on BirdSet benchmark, bird-specific training

Key question: Can these foundation models be distilled to CPU-viable inference while retaining bioacoustics performance?
