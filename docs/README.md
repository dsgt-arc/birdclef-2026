# Documentation

Shared context and documentation for the BirdCLEF 2026 discrete token pipeline research project.

## Structure

```
docs/
├── architecture/     # System design, alternative approaches, model architectures
├── experiments/      # Experiment results, logs, ablation studies
├── research/         # Literature reviews, research leads, notes
└── vendor/           # Documentation for vendored external repositories
```

### architecture/

Design documents for the pipeline and its components. Use this for:
- Alternative architectural approaches and trade-offs
- Data pipeline designs
- Model architecture decisions and rationale
- System diagrams

### experiments/

Experiment tracking and analysis. Use this for:
- Ablation study results
- Hyperparameter search logs
- Performance comparisons
- Training curves and metrics

### research/

Literature and exploration. Currently contains:
- `01_initial_research.md` - Initial literature review
- `02_leads.md` - Potential repositories and concepts to explore

### vendor/

Documentation for the 14 vendored repositories. See [vendor/README.md](vendor/README.md) for:
- Summary table of all repositories
- Pipeline stage mapping (tokenization, representation learning, embedding, inference)
- Category and status for each repository

## Quick Links

- [Vendor Repository Index](vendor/README.md)
- [Research Leads](research/02_leads.md)
- [Pipeline Proposal](../openspec/changes/add-discrete-token-pipeline/proposal.md)
