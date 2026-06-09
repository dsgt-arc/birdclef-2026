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
| `auto-motif-soundscape-efficacy.md` | Label-based extrinsic validation: production templates vs. `train_soundscapes_labels.csv`. Threshold-free ROC/PR-AUC + committed-threshold transfer (support-gated, calibrated); plus the window-level CFAR A/B that **falsifies** a label-free operating-point fix and shows calibration must be supervised. |
| `auto-motif-curation-annotation.md` | Manual-curation anchor (human ground truth): the Label Studio windowed-clip annotation workflow for 5 species in two role-split projects — calibrate d′→recovery, adjudicate the 74113 false positive, ground the strand-5 attractor diagnosis. Verdict protocol, scoring (`label_studio parse`), and the focal-precision interpretation boundary. |

## Concepts

`concepts/` holds research-idea documents (project convention).

| Concept | Idea | Status |
|---|---|---|
| `concepts/decouple-frequency-localization-from-focal-selection.md` | Mining motifs for the broadband/FM species the per-bin band-consensus gate is structurally blind to (~41% of soundscape∩train species). | Archived — canonical v1–v4 record + 5 validation strands. Decoupling premise falsified (v3); **v4** = the constructive resolution (recurrence-breadth gate + band-free clustering; n=17, Perch-embedding-corroborated qualitatively; the d′ ranking / per-species mode selector is *not* independently validated). **Strand 5** (label-free, D1): the strand-3 at-chance species are a *mining* failure (a non-specific broadband-energy detector, `auc_vs_class` ≤ 0.47, fixed cross-taxon attractor set) — not calibration, not domain shift — and it reaches the validated consensus route; 2/7 are genuine recoverable domain-shift. |
| `concepts/contrastive-matched-filter-exemplar-mining.md` | Constructive successor to strand 5: select the canonical template by a discriminative contrast — recurs within-species **and** far from a cross-species focal corpus (D1's negative) — computed on the existing 2-D matched filter; exemplar, not synthetic median. | Developing — design specified + two adversarial PI reviews; the matrix-profile *algorithm* rejected (mSTAMP's per-bin z-norm destroys the cross-band discriminant), re-grounded on the production `_score_track` z-NCC. Falsification: attractor re-mine + `22967` non-regression + strand-3 ground truth. Tracking: issue #49. |

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

`python -m data_analysis.auto_motif.focal_specificity` is the label-free
D1 probe (concept doc §v4 "strand 5"): the focal analog of the soundscape
ROC — the exact production matched-filter over focal audio of a species vs.
other species (folder = curation truth). `auc_vs_class` (vs same-class
focal) adjudicates the strand-3 at-chance species into a *mining* failure
(non-specific energy detector → discriminative mining / learned embedding)
vs. recoverable focal→soundscape domain shift. →
`artifacts/auto_motif/_focal_specificity/`.

`python -m data_analysis.auto_motif.soundscape_eval` is the label-based
extrinsic validation — drives the real production detector over the 66
labeled soundscapes and scores each species threshold-free (ROC/PR-AUC) and
at the committed threshold (strand 3). `--rescore cfar` adds the
window-level CFAR A/B (strand 4): it **falsifies** a label-free
operating-point fix (the optimal cut is prevalence-dependent → needs labels)
while proving the ROC ≥ 0.75 templates are operable, so calibration must be
supervised. See `auto-motif-soundscape-efficacy.md` and concept doc §v4
"v4 validation, strand 3"/"strand 4".

## Consolidation note

The Marbled White-lipped Frog files are intentionally split into:

- one cohesive case-study narrative, and
- two detailed supporting notebooks.

If this grows further, prefer keeping the case-study narrative as the entry point and moving low-level detail into supporting appendices or generated artifact summaries.
