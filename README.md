# BirdCLEF 2026

Research project exploring CPU-viable bioacoustic classification through foundation model distillation.

## Research Thesis

Foundation models like BEATs/AudioMAE learn superior bioacoustic representations but are excluded from BirdCLEF due to CPU inference constraints. This project explores strategies to unlock foundation model quality under CPU limits.

## Quick Start

| Resource | Description |
|----------|-------------|
| [docs/STATUS.md](docs/STATUS.md) | Current project state and next steps |
| [docs/concepts/](docs/concepts/) | Research ideas we're exploring |
| [docs/references/literature.md](docs/references/literature.md) | Key papers |

## Repository Structure

```
.
├── docs/                   # Documentation (start here)
│   ├── STATUS.md           # Current state summary
│   ├── concepts/           # Research ideas and hypotheses
│   ├── experiments/        # Experiment proposals and results
│   ├── references/         # Literature and deep research
│   ├── vendor/             # External tool documentation
│   └── _templates/         # Document templates
├── openspec/               # Change proposals (OpenSpec)
└── .claude/commands/       # Claude Code slash commands
```

## Documentation

See [docs/README.md](docs/README.md) for full documentation index.

### Key Concepts

| Concept | Description |
|---------|-------------|
| [soundscape-domain-shift](docs/concepts/soundscape-domain-shift.md) | Focal→soundscape domain gap |
| [data-and-compute-scale](docs/concepts/data-and-compute-scale.md) | Kaggle inference constraints |
| [mlp-mixer-distillation](docs/concepts/mlp-mixer-distillation.md) | CPU-efficient architectures |
| [research-leads](docs/concepts/research-leads.md) | EAT, OpenBEATs, BirdMAE |

## Current Focus

1. **EAT** - 15x more efficient than BEATs
2. **OpenBEATs** - SOTA on bioacoustics (300M params)
3. **BirdMAE** - Best on BirdSet benchmark

## Contributing

Use the slash commands to add documentation:
- `/docs:concept` - Add a research idea
- `/docs:experiment` - Propose an experiment
- `/docs:result` - Record experiment results
- `/docs:reference` - Capture external knowledge

See [AGENTS.md](AGENTS.md) for AI assistant guidelines.
