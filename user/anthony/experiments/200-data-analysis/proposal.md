# Proposal: Desktop Data Analysis Scratch Space

**Status**: proposed  
**Created**: 2026-05-09  
**Author**: Anthony

## Purpose

Use `user/anthony/experiments/200-data-analysis/` as a lightweight scratch space for interactive BirdCLEF 2026 analysis on the desktop workstation.

This is not a fixed hypothesis experiment. It is a container for small, focused analyses that answer follow-up questions quickly without waiting for SLURM.

## Data Sources

Primary data and large artifacts live outside the repo:

- BirdCLEF data mirror: `/mnt/data/scratch/birdclef-2026/`
- Metadata: `train.csv`, `taxonomy.csv`, `train_soundscapes_labels.csv`, `sample_submission.csv`
- Perch inference outputs: `/mnt/data/scratch/birdclef-2026/inference/perch-v2-tf/`
- Upstream experiment artifacts: especially `user/anthony/experiments/{100,106,107}/artifacts/`

See `CLAUDE.md` for exact paths and loading notes.

## Typical Questions

Examples of analyses this directory is meant to support:

- Perch embedding diagnostics and species clustering
- Soundscape label sanity checks across teachers/predictions
- Site / LOSO fold distribution shifts
- Per-species error analysis from downstream experiments
- Taxonomy, label, and metadata joins needed for modeling decisions

## Workflow

Each analysis should be small and self-contained:

1. State the question in a script/notebook header.
2. Load existing data/artifacts; do not re-extract large features here.
3. Save lightweight outputs to `artifacts/<topic>/`.
4. Add a short summary to `results.md` if the finding matters.
5. Spin out a numbered experiment if the analysis becomes a real hypothesis test.

## Success Criteria

This scratch space is useful if it produces figures/tables/notes that inform downstream modeling, evaluation, or postprocessing decisions.

Keep checked-in outputs lightweight: summary markdowns, figures, and small tables only. Do not copy large audio, embeddings, or parquet dumps into this directory.
