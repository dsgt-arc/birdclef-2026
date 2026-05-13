# Experiment 200 Data Analysis Notebooks

This directory uses short Markdown files as lightweight notebooks. Each notebook captures one analysis thread, with generated plots/tables kept under `../artifacts/`.

## Current notebooks

| Notebook | Purpose |
|---|---|
| `pacmap-soundscape-embeddings.md` | Site/species context structure in labeled soundscape Perch embeddings. |
| `labeled-species-training-coverage.md` | Counts train-audio files/windows for labels present in labeled soundscapes. |
| `representative-labeled-soundscape-species.md` | Defines the 10-species train-supported working set. |
| `selected-species-train-audio-pacmap.md` | Per-species train-audio PaCMAPs for the 10-species working set. |
| `marbled-white-lipped-frog-case-study.md` | Consolidated narrative for the Marbled White-lipped Frog analysis. |
| `marbled-white-lipped-frog-frequency-perch.md` | Detailed frequency-band and Perch-logit notes for the case study. |
| `marbled-white-lipped-frog-window-relevance.md` | Detailed weak relevance labeling notes for the case study. |
| `marbled-white-lipped-frog-motif-boxes.md` | Motif-box extraction workflow, what we tried, current curated annotations, and next steps. |
| `marbled-white-lipped-frog-motif-mining.md` | Candidate motif-box mining and clustering to avoid per-recording hand annotation. |
| `marbled-white-lipped-frog-auto-motif.md` | First run of the species-agnostic `auto_motif` pipeline on Marbled Frog; A/B vs the manual pipeline and known limitations. |

## Consolidation note

The Marbled White-lipped Frog files are intentionally split into:

- one cohesive case-study narrative, and
- two detailed supporting notebooks.

If this grows further, prefer keeping the case-study narrative as the entry point and moving low-level detail into supporting appendices or generated artifact summaries.
