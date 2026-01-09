<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

---

# Documentation Guidelines

## Adding Documentation

Use slash commands to create properly structured documents:

| Command | Purpose | Template |
|---------|---------|----------|
| `/docs:concept` | Research idea or hypothesis | `docs/_templates/concept.md` |
| `/docs:experiment` | Experiment proposal | `docs/_templates/experiment-proposal.md` |
| `/docs:result` | Experiment results | `docs/_templates/experiment-result.md` |
| `/docs:reference` | External knowledge capture | `docs/_templates/reference.md` |

## Documentation Structure

```
docs/
├── STATUS.md           # Current state (update when things change)
├── concepts/           # Research ideas (use /docs:concept)
├── experiments/        # Proposals + results (use /docs:experiment, /docs:result)
├── references/
│   ├── literature.md   # Paper links
│   └── deep-research/  # AI research outputs (use /docs:reference)
├── vendor/             # External tool docs
└── _templates/         # Templates (don't modify)
```

## Conventions

1. **Concepts**: Use kebab-case filenames, no date prefix (e.g., `soundscape-domain-shift.md`)
2. **Experiments**: Use `NNN-descriptive-name/` directories (e.g., `001-baseline-cnn/`)
3. **References**: Use `YYYY-MM-DD-descriptive-name.md` for deep-research
4. **Status**: Always update `docs/STATUS.md` after significant changes

## Workflow

```
Reference → Concept → Experiment Proposal → Results
    ↓          ↓              ↓                ↓
 literature  ideas      test hypothesis    document
```

When you learn something new:
1. Capture in `references/` if external knowledge
2. Add to `concepts/` if it's a testable idea
3. Create experiment when ready to test
4. Update `STATUS.md` to reflect current state
