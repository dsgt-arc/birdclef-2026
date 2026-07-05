# BirdCLEF 2026

DS@GT ARC BirdCLEF+ 2026 code, experiment logs, and project documentation.

The competition phase is complete. The current write-up is the CEUR working note **“Can Tokens Compete? Token Representations against Supervised CNN Backbones for BirdCLEF+ 2026”**.

## Current result

Best submitted system: **0.931 public / 0.936 private LB**.

That system ensembles:

1. frozen **Perch v2** embeddings with MLP probes,
2. **HGNetV2-B0 SED** on log-mel spectrograms, and
3. a lightweight non-bird prototypical head over Perch embeddings.

The paper also reports token/foundation-model findings: WavTokenizer codec tokens failed the bioacoustic retrieval gates; frozen Perch transferred best among tested encoders; BEATs/EAT/AST/SSAMBA did not close the gap to the specialist Perch-backed system.

## Start here

| Resource | Purpose |
|---|---|
| [docs/STATUS.md](docs/STATUS.md) | Current project state and headline results |
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/concepts/working-note.md](docs/concepts/working-note.md) | Paper framing and result map |
| [AGENTS.md](AGENTS.md) | Assistant/project operating rules, especially PACE venv policy |

## Repository layout

```text
.
├── .claude/               # Claude slash commands and project skills
├── docs/                  # Curated project documentation
├── user/                  # Per-user code and experiment workspaces
│   ├── anthony/experiments/
│   ├── murilo/experiments/
│   └── adrian/experiments/
├── vendor/                # Vendored external code/submodules
└── AGENTS.md              # Repo-specific agent instructions
```

## Notes for future work

Prefer documenting durable findings in:

- experiment `results.md` files under `user/<name>/experiments/`,
- [docs/STATUS.md](docs/STATUS.md) for project-level state,
- [docs/concepts/README.md](docs/concepts/README.md) for status-indexed research ideas, and
- the working-note draft for publication text.
