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
| `auto-motif-pipeline.md` | Species-agnostic reference: pipeline stages, config knobs (incl. the recurrence-breadth gate, `band_mode`/`cluster_mode`), output schema, and the band-discovery genericity-boundary resolution. |
| `auto-motif-soundscape-efficacy.md` | Label-based extrinsic validation: production templates vs. `train_soundscapes_labels.csv`. Threshold-free ROC/PR-AUC + committed-threshold transfer, support-gated and calibrated. |

## Concepts

`concepts/` holds research-idea documents (project convention).

| Concept | Idea | Status |
|---|---|---|
| `concepts/decouple-frequency-localization-from-focal-selection.md` | Mining motifs for the broadband/FM species the per-bin band-consensus gate is structurally blind to (~41% of soundscape∩train species). | Archived — canonical v1–v4 record. Decoupling premise falsified (v3); **v4** = the constructive resolution (recurrence-breadth gate + band-free clustering; n=17, Perch-embedding-corroborated qualitatively; the d′ ranking / per-species mode selector is *not* independently validated). |

## Production routing

`python -m data_analysis.auto_motif.production_manifest` consolidates the
already-materialized batch artifacts (no recompute) into
`artifacts/auto_motif/_production_manifest.csv` — the canonical per-species
routing table for the full 41-species soundscape∩train subset (24 `consensus`,
17 `consensus_blind`→`free+global`, `gabor_stack` opportunistic). Rationale and
the conservative-routing argument: concept doc §v4 + "v4 validation".

`python -m data_analysis.auto_motif.cofire_fragmentation` is the label-free
fragmentation guardrail — `frac_distinct` per species (co-firing collapse, no
labels/Perch) flags where a mode's family richness is spurious
over-segmentation. Rationale: concept doc §v4 "v4 validation, strand 2".

`python -m data_analysis.auto_motif.soundscape_eval` is the label-based
extrinsic validation — drives the real production detector over the 66
labeled soundscapes and scores each species threshold-free (ROC/PR-AUC) and
at the committed threshold. Calibrated finding (support-gated): see
`auto-motif-soundscape-efficacy.md` and concept doc §v4 "v4 validation,
strand 3".

## Consolidation note

The Marbled White-lipped Frog files are intentionally split into:

- one cohesive case-study narrative, and
- two detailed supporting notebooks.

If this grows further, prefer keeping the case-study narrative as the entry point and moving low-level detail into supporting appendices or generated artifact summaries.
