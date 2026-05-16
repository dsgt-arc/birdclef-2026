# Concept: Decouple frequency localization from focal-species selection

**Status**: archived (explored v1/v2/v3, concluded negative — the
decoupling premise is structurally flawed; see v3 result)
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
ROI with shape features, and cluster them with **frequency as a soft
prior, not a gate** — frequency centroid/extent is a clustering *feature*
(`maad.features.centroid_features`), optionally via a multi-resolution
filterbank (`maad.features.filter_multires`); it informs grouping but can
never return "zero bands" and abort. Re-implement focal-species selection
(job 2) as **how well a cluster's template applies across the focal
training set**: keep clusters whose template recurs well over the
species' focal recordings; refine each with the existing EM stage; detect
with `maad.rois.template_matching`.

**Why job 2 is largely free.** Focal `train_audio/<label>/` recordings
are *species-curated* — the labeled species is by construction the one
thing common across the set. The focal prior is therefore **intrinsic to
the data**, not something to reconstruct with a background corpus. A
heterospecific appears in *some* recordings idiosyncratically; the focal
species recurs across *most* of them, so a "template fits broadly across
the focal set" criterion (the existing `member_bg`/recurrence machinery
on the shape axis) selects it without an explicit negative set. The
narrowband assumption is removed; focal specificity is preserved by the
data curation plus shape-axis recurrence.

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

- **Focal selection is largely given by the data; the residual is
  systematic co-occurrence.** Because the focal set is species-curated,
  v1 selection = "template fits broadly across the focal recordings"
  (within-focal recurrence) — no background corpus needed. The honest
  residual: focal recordings are species-*primary*, not species-*pure*.
  A heterospecific that co-occurs *systematically* across many of this
  species' recordings (shared habitat/soundscape) would also fit broadly
  and survive. That tail (rarer than generic background) is the only case
  needing a cross-species discriminative check — a *discriminative*
  recurrence (this-species vs other-species corpus, TF-IDF-like) or a
  Perch-embedding focal prior. Defer to v2; quantify the contamination
  first.
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
3. **Within-focal template-fit suffices for selection.** Selecting shape
   clusters purely by template-fit/recurrence across the *focal set* (no
   background corpus) recovers the species' motif, and the residual
   contamination — fraction of selected clusters that are heterospecific
   (audited on Marbled Frog vs its curated boxes, and spot-checked on a
   few recovered FM species) — is small (target < ~15%), confirming the
   species-curation prior carries job 2.

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

## v1 result (2026-05-16)

Minimal toggle implemented: `AutoConfig.band_free` / `--band-free` takes
WIDEBAND as one band, reusing all downstream stages (commit `dac1e89`).
A/B (band-free vs band-gate):

| species | band-gate | band-free |
|---|---|---|
| `22967` Marbled Frog (narrowband, **multi-motif**) | 5 fam, curated rec **0.97**, d′3.75 | 1 fam, rec **0.36**, d′1.28 |
| `23158` Pale-legged Weeping Frog (gate: **0 bands, hard fail**) | — | **1 fam, d′3.14, AUC0.98** |
| `hyamac1` Hyacinth Macaw (gate: **0 bands, hard fail**) | — | **1 fam, d′4.01, AUC0.99** |

**Confirmed:** Hypothesis 1 directionally holds — band-free + EM +
`member_bg` recovers strong, well-separated motifs for FM/broadband
species the gate hard-failed. Segmentation + selection mechanically work.

**Falsified:** Hypothesis 2 in the *minimal* form — band-free regresses
multi-motif narrowband species. Mechanism (evidenced, not assumed): the
band-gate's 5 families came from 5 *bands* (~1 shape-cluster each); the
frequency split was carrying the multi-motif separation. Band-free puts
all 1006 ROIs in one band; HDBSCAN over Gabor shape features finds only
**2** clusters (≈0 noise) → 1 family. The frog's chip/primary/secondary
are **shape-similar but frequency-distinct**; centroid-frequency as 1 of
26 Gabor features is too weak a prior to split them.

**Refined next lever:** "frequency as a soft *centroid feature*" is
empirically too weak. It must be a stronger *stratifying* prior without
reverting to a hard gate. Candidates: (a) over-segment — frequency-
stratified clustering (cluster within coarse overlapping freq strata) or
up-weight the centroid feature / lower `shape_cluster_min_size` — then
let job-2 (template-fit-across-focal recurrence + `member_bg`) keep the
good clusters rather than trusting one global HDBSCAN; (b) per-frequency
adaptive ROI thresholding (the global wideband percentile lets loud
broadband drown faint narrowband). This keeps job-1 band-free while
restoring multi-motif resolution.

## v2 result (2026-05-16): Gabor front-end explored — not better

Pivoted to test an FM-aware *detector* (job-1): `band_mode="gabor"` —
band-free ROIs from a maad oriented 2D-Gabor-filterbank energy map
(`filter_bank_2d_nodc`/`filter_multires`), commits `022684f`/`e10961c`.
A/B vs band-gate and band-free intensity:

| species | band-gate | band-free (intensity) | gabor |
|---|---|---|---|
| `22967` frog (regression gate) | **0.97**, 5 fam | 0.36, 1 fam | **0.00**, 1 fam |
| `nacnig1` narrowband bird | 5 fam, d′→3.49 | — | 1 fam, d′2.67 |
| `23158` (gate hard-failed) | — | **d′3.14** | **0 families (hard fail)** |
| `hyamac1` (gate hard-failed) | — | **d′4.01** | d′3.45 |

**Conclusion: the Gabor detector is the *worst* of the three.** It
under-detects short/narrowband calls (frog 0.00; `23158` yields no
families where band-free gets d′3.14) and even on the broadband macaw
band-free intensity beats it (4.01 > 3.45). The conceptual case for B
(structure-matched, faint-call sensitivity) didn't survive data: these
calls are loud enough that intensity already catches them, and the maad
preset under-responds to short narrowband chips. **Decisive finding: the
bottleneck across *both* non-consensus modes is job-2 (clustering
collapse — one global HDBSCAN over one band collapses multi-motif
species), not job-1 (detection). Swapping the detector cannot fix it.**

**Recommended direction (unchanged from the v1 lever): frequency-
stratified soft clustering (A).** Gabor is recorded as an
explored-and-rejected branch; the toggle stays in the code for
reproducibility but is *not* preferred. **Consensus remains the
production default** — it is the only mode that does not regress the
validated Marbled Frog; band-free/gabor are opt-in experiments. No prior
implementation is downweighted in favor of Gabor (the data forbids it).

## v3 result (2026-05-16): freq-stratified clustering explored — regresses the decisive test; decoupling premise is structurally flawed

Implemented lever A — `cluster_mode="freq_stratified"` (commit `1039b3c`):
in non-consensus modes the single WIDEBAND band is sub-divided into
data-derived frequency strata (loose-ROI centre-freq density valleys),
which become the bands for the existing per-band HDBSCAN + EM, so
consensus's per-band separation is reproduced *without a hard gate*.

The strata smoothing scale is **one median call-width**, not an
n-adaptive statistical rule. This was forced empirically: Scott KDE,
GMM+BIC, HDBSCAN-1D, Hartigan's dip test, and a morphology-scaled KDE
were each tried and **all over-segment the broadband macaw at n≈20k** —
its centre-freq density genuinely ripples at the narrowband frog's mode
spacing, so any n-adaptive rule manufactures spurious strata. (Every
fixed bandwidth is itself a tuned constant; the call-width scale is the
one physically-motivated, data-derived choice — two motifs closer than a
call's own frequency extent cannot be separated by *any* horizontal
split.)

A/B (band-free detector held fixed; only the clustering differs):

| species | free+global | free+gabor | free+strat |
|---|---|---|---|
| `22967` frog (regression test) | 0.36 rec, 1 fam, d′1.28 | 0.00 rec | **0.00 rec**, 5 fam, d′9.74† |
| `hyamac1` (gate hard-failed) | **d′4.01**, 1 fam | d′3.45 | d′3.44, 8 fam |
| `23158` (gate hard-failed) | **d′3.14**, 1 fam | 0 fam | d′2.90, 6 fam |
| `nacnig1` narrowband bird | d′2.85, 2 fam | d′2.67 | d′2.98, 8 fam (consensus: **d′3.49**) |

†a degenerate 7-peak overfit shard (AUC=1.0), not quality.

**Verified mechanism (not an eval artifact).** On the frog reference file
(8 curated calls at 1.1–3.8 s) the stratified run emits only **2
consolidated events, at 4.0 s and 14.2 s** — genuinely no temporal
overlap with any curated call (`time_iou=0` is correct). The
data-derived stratum boundaries (200 / **1789** / 3843 / 4978 / 6000 Hz)
**bisect the frog's coherent motif** (primary ~680–1300 Hz, secondary
~1500–2140 Hz): the [200,1789] stratum template has 107 peaks but
**d′0.046** (clipping off the secondary destroyed its distinctiveness),
and the [1789,3843] strata are tiny 5–7-peak overfit shards. The single
recoverable motif band-free captures as one 56-peak family is shredded
into one garbage template + two shards that detect almost nothing.

**The cross-stratum merge (the "+ merge" half of this concept) is moot.**
It would re-group per-stratum templates that are *already degraded* by
motif-incoherent freq-clipping; re-merging a garbage template and overfit
shards cannot reconstruct the destroyed time-frequency gestalt. The
failure is "coherent motif bisected into bad pieces," not "good motif
over-split into similar pieces" — the merge addresses the wrong problem.

**This falsifies the v1/v2 "recommended direction" and the concept's core
premise.** The premise was that frequency localization and focal
selection are *separable* — that re-deriving frequency localization from a
different statistic, freed of the recurrence consensus, would mine
broadband/FM motifs while preserving the narrowband ones. The data shows
they are **not separable**: the consensus band split worked for the frog
(0.97) *specifically because* per-bin cross-recording recurrence
consensus aligns bands to the actual repeated call structure. A
horizontal partition derived from anything else (centre-freq KDE here;
oriented-Gabor energy in v2) shreds coherent motifs. The only frequency
localizer that aligns to motif identity *is* the recurrence consensus —
the very statistic that is structurally blind to broadband/FM. They are
the same mechanism; you cannot decouple them.

**Status: archived (explored, concluded negative).** Consensus remains
the production default — it is the only mode that does not regress the
validated Marbled Frog (free/gabor/strat all leave it ≤0.36 vs 0.97).
`cluster_mode` stays in the code for reproducibility, default `global`,
not preferred. The forward implication — that motif-mining is
intrinsically a narrowband-tonal method and broadband/FM species should
be routed to learned embeddings (Perch/BirdNET, already in the project)
rather than to a patched miner — is flagged as an **open
research-direction question for the user**, not adopted here.

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
- *Naive `create_mask` on the wideband spectrogram, no selection at all*
  — rejected: keeping *every* ROI mines heterospecifics. But the
  selection that replaces the consensus need not be a background-corpus
  discriminator: because the focal set is species-curated,
  within-focal-set template-fit/recurrence is sufficient for v1. (This
  refines the earlier over-cautious "naive band-free loses focal
  specificity" — it loses it only if you also drop the recurrence/fit
  selector; the curated focal set, not the per-bin consensus, is what
  supplies the prior.)
- *Full learned-embedding (Perch) detector* — strong focal prior but
  heavier and less interpretable than the shape-cluster path; keep as a
  fallback or as the discriminative-recurrence signal in (c).

**Phasing.** Keep the band-gate path as default. Add band-free as an
opt-in toggle so it can be A/B'd on the existing 41-species harness
without risking the validated Marbled Frog result. Promote to default
only if Hypotheses 1 & 2 both hold.
