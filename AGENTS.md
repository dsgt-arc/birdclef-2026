# OpenSpec (OPSX Workflow)

This project uses [OpenSpec](https://github.com/Fission-AI/OpenSpec) for spec-driven change management. The OPSX workflow replaces the legacy phase-locked approach with fluid, iterative actions.

When a request involves architecture changes, new capabilities, breaking changes, or big decisions:
1. Use `/openspec:proposal` to create a change proposal
2. Use `/openspec:apply` to implement approved changes
3. Use `/openspec:archive` to finalize completed work

Configuration: `openspec/project.md`
Changes: `openspec/changes/<name>/`

See `openspec/AGENTS.md` for the full workflow and spec format.

---

# Documentation Guidelines

## Adding Documentation

Use slash commands to create properly structured documents:

| Command | Purpose | Template |
|---------|---------|----------|
| `/docs:concept` | Research idea or hypothesis | `docs/_templates/concept.md` |
| `/docs:reference` | External knowledge capture | `docs/_templates/reference.md` |
| `/docs:vendor` | External tool/dataset documentation | `docs/vendor/_TEMPLATE.md` |
| `/experiment:proposal` | Experiment proposal | `docs/_templates/experiment-proposal.md` |
| `/experiment:result` | Experiment results | `docs/_templates/experiment-result.md` |
| `/experiment:validate` | Validate experiment structure | - |
| `/experiment:summary` | Experiment status summary for session context | - |

## Documentation Structure

```
docs/
├── STATUS.md           # Current state (update when things change)
├── concepts/           # Research ideas (use /docs:concept)
├── references/
│   ├── literature.md   # Paper links
│   └── deep-research/  # AI research outputs (use /docs:reference)
├── vendor/             # External tool docs (use /docs:vendor)
└── _templates/         # Templates (don't modify)

user/<name>/experiments/ # Experiments per user (use /experiment:*)
```

## Conventions

1. **Concepts**: Use kebab-case filenames (e.g., `soundscape-domain-shift.md`)
2. **References**: Use `YYYY-MM-DD-descriptive-name.md` for deep-research
3. **Vendor**: Use kebab-case filenames (e.g., `birdnet-analyzer.md`)
4. **Experiments**: Use `NNN-descriptive-name/` directories in `user/<name>/experiments/`
5. **Status**: Always update `docs/STATUS.md` after significant changes

## Workflow

```
Reference → Concept → Experiment Proposal → Results
    ↓          ↓              ↓                ↓
 literature  ideas      test hypothesis    document
```

When you learn something new:
1. Capture in `references/` if external knowledge
2. Add to `concepts/` if it's a testable idea
3. Create experiment when ready to test (`/experiment:proposal`)
4. Validate before running (`/experiment:validate`)
5. Document results (`/experiment:result`)
6. Update `STATUS.md` to reflect current state

Experiments are tracked per-user in `user/<name>/experiments/`.
