# Documentation

Curated documentation for the DS@GT ARC BirdCLEF+ 2026 repository.

The docs are organized so current conclusions are easy to find without deleting the exploratory trail. Concept notes may be current, historical, speculative, or negative-result records; see the concept index for status.

## Start here

| Document | Purpose |
|---|---|
| [STATUS.md](STATUS.md) | Current results, paper state, and what matters now |
| [concepts/working-note.md](concepts/working-note.md) | Working-note framing and result map |
| [operations.md](operations.md) | Operational notes for running jobs and managing artifacts |
| [references/literature.md](references/literature.md) | Key papers and external references |
| [vendor/README.md](vendor/README.md) | Index of vendored external projects |

## Current narrative

The project no longer centers on the original “distill foundation models into a CPU-viable student” thesis. The current paper has two pillars:

1. **Competitive supervised system** — Perch v2 probes + HGNetV2-B0 SED + a non-bird prototypical head reach **0.931 public / 0.936 private LB**.
2. **Token/foundation-model comparison** — codec tokens, semantic audio encoders, and SSAMBA-style sequence models are evaluated against that supervised baseline.

Headline finding: token/foundation encoders are useful research objects, but the strongest deployed system is still a practical specialist ensemble built around Perch and a supervised SED backbone.

## Concept docs

See [concepts/README.md](concepts/README.md) for a status-indexed concept map:

- current anchors,
- historical/superseded notes,
- exploratory future-work seeds,
- falsified or shelved threads.

Detailed experiment evidence lives under `user/<name>/experiments/*/results.md`; concept notes capture framing and provenance.

## References

- [references/literature.md](references/literature.md) — papers and systems that inform the write-up
- [references/kaggle/](references/kaggle/) — Kaggle notebook/discussion summaries
- [references/deep-research/](references/deep-research/) — older AI-generated research syntheses; keep as background, not source of truth
- [vendor/](vendor/) — tool and model documentation for vendored projects
