# Concept: Decouple frequency localization from focal-species selection

**Status**: developing
**Created**: 2026-05-16
**Updated**: 2026-05-16

## Idea

Stage-2 band discovery in the auto-motif pipeline silently does **two**
jobs at once:

1. **Frequency localization** — find the frequency band(s) carrying the
   motif (a per-frequency-bin statistic).
2. **Focal-species selection** — keep only energy that belongs to the
   labeled species, by requiring it to **recur across the species'
   recordings** (the cross-recording median consensus suppresses
   heterospecifics and noise that don't recur).

These are conflated onto a single axis: *recurrence of energy in a
frequency bin*. That statistic only concentrates for **narrowband,
frequency-stable, stereotyped** signals. Broadband and FM-sweep calls
spread their energy across bins (and the bins shift between renditions),
so job 1 fails — and because the two jobs are coupled, the whole run
fails (zero bands).

The proposal: **separate the two axes.** Do *band-free* 2D ROI
segmentation with `maad.rois` across all recordings (job 1 becomes
image-domain segmentation that handles any call shape), describe every
ROI with shape features, and re-implement focal-species selection (job 2)
as **recurrence of ROI *shape clusters* across the species' recordings**
rather than recurrence of frequency bins. Recurrent shape clusters are
the focal motif families; refine each with the existing EM stage; detect
with `maad.rois.template_matching`. The narrowband assumption is removed;
focal-species specificity is preserved by moving it to a shape-invariant
axis.

## Motivation

The 2026-05-16 Luigi batch over the 41 soundscape∩train species
(`min_focal=8`): **24 ok / 17 failed, every failure the identical**
`Auto band discovery returned no bands` (no crashes). Failures cluster on
broadband / harmonic / FM-sweep callers; successes are narrowband/tonal
repeaters. It spans all taxa and all `n_focal` (10→315) — a structural
limit, not a small-sample artifact (`artifacts/auto_motif/_batch/manifest.csv`).

This is most of the target, not an edge case: **12 of 17 failures are
birds**, and birds (FM-rich) dominate the labeled soundscapes. A
species-agnostic motif miner that is blind to bird song is not
species-agnostic.

The fix is cheap in surface area: the downstream stages
(`maad.rois.create_mask`/`select_rois` ROI proposal → 26-D Gabor
`shape_features` → HDBSCAN shape clustering → EM template →
`maad.rois.template_matching` detection) are **already shape-agnostic**,
and `maad.rois` is **already a dependency**. Only the stage-2 gate is
narrowband.

## Background

- `docs/auto-motif-pipeline.md` — §2 "Band discovery" (the
  structural-assumption callout), §"Library choices" (records that band
  discovery is the deliberately hand-rolled piece because no off-the-shelf
  detector produces "frequency bands where this species recurs across
  recordings"), and the lead bullet in §"Limitations and follow-ups"
  (the batch evidence and library-grounded fix direction).
- `maad.rois.create_mask` / `select_rois` / `rois_2d` /
  `spectrogram_local_max` — image-domain spectrogram segmentation
  (Towsey-style); shape-agnostic by construction.
- `maad.rois.template_matching` — normalized spectrogram
  cross-correlation = the Mellinger & Clark (2000) detector, the standard
  literature method for FM moans/sweeps. Already used in stage 7.
- `maad.features.shape_features` (2D Gabor filterbank,
  `opt_shape_presets`) + `centroid_features` — describe arbitrary ROI
  morphology; already used in stage 5.
- Multiple-instance / recurrence framing: the existing exemplar-weighting
  and the `member_bg_dprime`/`auc` label-free quality screen are the same
  "what recurs in this species vs background" idea — reusable on the
  shape axis.
- opensoundscape's own answer for arbitrary call structure is learned
  models (`opensoundscape.CNN`, BirdNET/Perch). Perch embeddings already
  exist for this dataset — a candidate auxiliary focal prior.

## Open Questions

- **The crux: focal-species selection without a frequency prior.**
  Band-free segmentation across a soundscape returns ROIs for *every*
  sound. What makes a shape cluster "the focal species"? Candidates:
  (a) shape clusters that recur across many of the species' recordings;
  (b) *discriminative* recurrence — recur in this species' recordings
  more than in a background/other-species corpus (TF-IDF-like, or the
  existing member-vs-background screen lifted to the shape axis);
  (c) a Perch-embedding focal prior. A co-occurring background species
  present across many focal recordings would falsely recur under (a)
  alone — (b)/(c) are the mitigation.
- **Shape-descriptor invariance.** Do Gabor `shape_features` place
  FM sweeps of differing rate/extent/onset in one cluster, or fragment
  them? Need time/frequency-translation invariance but contour-shape
  sensitivity. Which `opt_shape_presets` resolution?
- **Clustering at corpus scale.** HDBSCAN over all ROIs across all
  recordings of a species could be 10⁴–10⁶ ROIs for n≈500. Cost, memory,
  parameter stability, subsampling strategy.
- **EM without a band.** The current EM anchors via the band-mean
  envelope onset (`_refine_anchor_via_envelope`) — band-dependent. A
  band-free path needs a different anchor (template-centroid or
  cross-correlation-peak alignment).
- **Precision / thresholding.** Still no labeled negatives; reuse the
  member-vs-background d′/AUC screen as the label-free quality gate.
- **Non-regression.** Does shape-axis recurrence stay as focal-specific
  as frequency-axis recurrence for the narrowband species that currently
  work (esp. Marbled Frog)?

## Testable Hypotheses

1. **Recovery.** Band-free `maad.rois` segmentation + shape-cluster
   recurrence yields ≥1 high-quality motif family (member-vs-background
   d′ ≥ 2.0) for a *majority* of the 17 currently-failing species, with
   zero crashes — measured by re-running the existing Luigi batch with a
   `band_free` toggle and diffing the manifest.
2. **Non-regression.** On the 24 currently-OK species, the band-free path
   does not degrade the label-free quality screen, and Marbled Frog
   (22967) curated recovery stays ≥ 0.90 — i.e., shape-axis recurrence is
   at least as focal-specific as frequency-axis recurrence for narrowband
   species.
3. **Discriminative recurrence controls heterospecifics.** Selecting
   shape clusters by *discriminative* recurrence (this-species vs a
   background corpus) yields higher member-vs-background d′ than raw
   per-cluster frequency-of-occurrence — i.e., the false-focal-cluster
   risk is controllable without a frequency prior.

## Connections

- **Based on**: `docs/auto-motif-pipeline.md` (§2, Library choices,
  Limitations); the band-discovery genericity-boundary finding
  (`artifacts/auto_motif/_batch/manifest.csv`, 24/41).
- **Related to**: the member-vs-background separation screen and
  exemplar-weighting (both are "what recurs in this species vs
  background", reusable on the shape axis); Perch embeddings as an
  auxiliary focal prior.
- **Could lead to**: an experiment proposal — A/B the narrowband
  band-gate vs a band-free shape-recurrence path on the 41-species Luigi
  batch (reusing `data_analysis.auto_motif.batch` and the manifest),
  gated behind an opt-in `AutoConfig` toggle so the validated frog path
  stays default until parity is shown.

## Notes

**Reuse map.** Unchanged: `maad.rois.create_mask`/`select_rois`,
`maad.features.shape_features`/`centroid_features`, HDBSCAN shape
clustering, the EM loop, `maad.rois.template_matching` detection, the
Luigi driver + manifest, the member-vs-background screen, NMS +
consolidation. Changed: stage 2 (per-bin consensus) → corpus-wide
band-free segmentation + cross-recording **shape-cluster** recurrence as
the focal filter. New: a band-free EM anchor; a discriminative-recurrence
focal selector.

**Alternatives considered.**
- *Lower `k_band_mad`* — rejected: a tuned-constant hack that still
  doesn't address FM/broadband per-bin washout.
- *Naive `create_mask` on the wideband spectrogram, no consensus* —
  rejected: removes the narrowband assumption but **loses focal-species
  selection** (mines every heterospecific). This is the self-correction
  to the first-pass "just swap stage 2 to maad.rois" idea; the consensus
  is doing real work and must be replaced, not deleted.
- *Full learned-embedding (Perch) detector* — strong focal prior but
  heavier and less interpretable than the shape-cluster path; keep as a
  fallback or as the discriminative-recurrence signal in (c).

**Phasing.** Keep the band-gate path as default. Add band-free as an
opt-in toggle so it can be A/B'd on the existing 41-species harness
without risking the validated Marbled Frog result. Promote to default
only if Hypotheses 1 & 2 both hold.
