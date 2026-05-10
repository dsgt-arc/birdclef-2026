# Proposal: Desktop Data Analysis Scratch Space

**Status**: proposed
**Created**: 2026-05-09
**Author**: Anthony

## Hypothesis

Running ad-hoc data analyses on the desktop workstation (rather than PACE) will accelerate exploratory iteration where the bottleneck is interactive turnaround, not compute. Analyses that would otherwise stall waiting for a SLURM allocation can be unblocked by working against the local mirror at `/mnt/data/scratch/`.

## Background

Experiment 100 (`100-eda`) produced the canonical EDA artifacts on PACE — label mapping, taxonomy coverage, class distribution, soundscape label structure. Since then, follow-up questions have surfaced from exps 102/106/107:

- **Embedding diagnostics** — ridge vs hopfield divergence on specific species; which Perch v2 embeddings cluster cleanly vs collapse?
- **Soundscape label sanity checks** — how do teacher predictions (BirdNET, Perch, BamBird) agree/disagree on the 66 labeled files?
- **LOSO fold structure** — per-site distribution shifts that explain the 0.749 LOSO → 0.804 LB gap from exp 106.
- **Per-species error analysis** — where exp 102/106 fail; which species are amphibian/insect vs avian; what's the taxa-conditional error rate?

These are interactive analyses better suited to a Jupyter / matplotlib loop on the desktop than a batch SLURM job. The desktop has the same scratch tree as PACE (`/mnt/data/scratch/` ↔ `~/scratch/`) so artifacts and embeddings are already in place.

This experiment is a **container** for those ad-hoc analyses. Each notebook or script under `notebooks/` answers one focused question and ships its findings to `artifacts/` (figures, parquets, summary markdowns). Sub-analyses get tracked in `tasks.md` as they accrue.

## Method

### Approach

Lightweight notebook-driven analysis. No training, no Luigi pipeline. Each analysis is a single notebook or script that:

1. Loads existing artifacts from PACE-mirrored scratch paths or upstream experiments (100, 106, 107)
2. Produces a focused result — a figure, a parquet, a short summary
3. Is self-contained enough that future-me can re-run it without spelunking

### Setup

| Component | Details |
|-----------|---------|
| Data | `/mnt/data/scratch/birdclef-2026/` (mirror of PACE `~/scratch/`); upstream artifacts in `user/anthony/experiments/{100,106,107}/artifacts/` |
| Compute | Desktop workstation (no SLURM); Python interactive |
| Dependencies | polars, duckdb, matplotlib, seaborn, scikit-learn, umap-learn (for embedding plots); shared deps in `user/anthony/experiments/106-perch-bayesian/` |
| Code | `user/anthony/experiments/200-data-analysis/` |
| Outputs | `artifacts/` — figures, parquets, summary markdowns |

### Procedure

Each sub-analysis follows the same shape:

1. State the question in a notebook header or script docstring
2. Load inputs from existing artifacts (do not re-derive)
3. Produce the smallest figure / table that answers the question
4. Save to `artifacts/<topic>/` and add a one-paragraph summary to `results.md`
5. Tick the corresponding `tasks.md` entry

### Variables

- **Independent**: which dataset slice / experiment / species cohort the analysis targets
- **Dependent**: whatever metric or distribution the analysis surfaces
- **Controlled**: input artifacts are frozen — no re-extraction inside this experiment

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Analyses delivered | count of completed sub-analyses with figures + writeup in `artifacts/` |
| Decisions informed | count of sub-analyses cited in a downstream experiment proposal or result |

### Baseline

No baseline — this is exploratory.

### Success Criteria

- **Confirm if**: at least one sub-analysis produces a finding that changes a decision in another experiment (modeling, postprocessing, eval)
- **Reject if**: N/A — analyses are always informative even when null

## Limitations

- Desktop is single-user and not backed up the same way PACE is — artifacts that matter long-term should be checked into git or copied to PACE
- This experiment intentionally has open scope; resist the urge to grow it indefinitely. Spin out a numbered experiment when an analysis evolves into a hypothesis test.
- Embedding artifacts (Perch v2, etc.) are large — keep `artifacts/` to summary tables and figures, not raw arrays
