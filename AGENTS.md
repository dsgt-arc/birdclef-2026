# CRITICAL: Virtual Environment Policy (PACE Cluster)

**NEVER install `.venv` under the home directory or any subdirectory of this project.**
The home directory on PACE has a **20GB quota**. Installing venvs there WILL cause OOM,
crash the session, and potentially corrupt work.

**Interactive sessions** — use `$TMPDIR` (SLURM) or `/tmp/` (fallback):
```bash
# Convention: $TMPDIR/birdclef-<JOB_ID>/.venv
VENV_BASE="${TMPDIR:-/tmp}"
JOB_ID="${SLURM_JOB_ID:-$$}"
JOB_VENV_DIR="$VENV_BASE/birdclef-$JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"
uv pip install -e "$EXPERIMENT_DIR"
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**SLURM batch jobs** — always use `$TMPDIR` (fast local NVMe):
```bash
JOB_VENV_DIR="$TMPDIR/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"
```

**NEVER use bare `pip install`** — it writes to `~/.local/` and eats home dir quota:
```bash
pip install nnAudio                           # WRONG — writes to ~/.local (7GB+ for torch)
pip install --user some-package               # WRONG — same thing
```
**Always use `uv pip install` inside an activated venv:**
```bash
source "$JOB_VENV/bin/activate"
uv pip install -e ".[dev]"                    # CORRECT — installs into venv
uv pip install nnAudio                        # CORRECT — installs into venv
```

**NEVER use `uv sync`** — it ignores the active venv and creates `.venv` in cwd:
```bash
source "$JOB_VENV/bin/activate"
cd user/murilo/experiments/005-*/
uv sync                                      # WRONG — creates .venv in cwd (home dir), ignores active venv
uv pip install -e ".[dev]"                   # CORRECT — installs into active venv
```

**WRONG — never do this:**
```bash
python -m venv .venv                          # pollutes home dir
uv venv user/murilo/experiments/004-*/.venv   # same problem
uv sync                                      # creates .venv in cwd, ignores active venv
```

See `user/murilo/create_venv.sh` and `user/murilo/experiments/003-model-training/sbatch/_job.sh` for reference.

---

# Documentation Guidelines

## Adding Documentation

Use slash commands to create properly structured documents:

| Command | Purpose | Template |
|---------|---------|----------|
| `/docs:concept` | Research idea or hypothesis | `docs/_templates/concept.md` |
| `/docs:reference` | External knowledge capture | `docs/_templates/reference.md` |
| `/docs:vendor` | External tool/dataset documentation | `docs/vendor/_TEMPLATE.md` |
| `/docs:kaggle-summary` | Clean-room Kaggle notebook/dataset summary | `docs/_templates/kaggle-summary.md` |
| `/kaggle:vendor` | Vendor Kaggle notebooks/datasets into `vendor/kaggle/` | `.kaggle-datasets` |
| `/experiment:proposal` | Experiment proposal | `docs/_templates/experiment-proposal.md` |
| `/experiment:result` | Experiment results | `docs/_templates/experiment-result.md` |
| `/experiment:validate` | Validate experiment structure | - |
| `/experiment:summary` | Experiment status summary for session context | - |
| `/experiment:context` | Full experiment context loader for session start | - |
| `/kaggle:discussion` | Fetch + document a Kaggle discussion post | `docs/references/kaggle/` |
| `/kaggle:notebook` | Pull + analyze a Kaggle notebook | - |

## Tooling

Shared helper scripts now live with the experiment that uses them (for example under `user/<name>/experiments/<id>/scripts/`). The old root-level `scripts/` directory has been removed.

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
