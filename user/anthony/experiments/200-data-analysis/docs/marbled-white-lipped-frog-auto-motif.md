# Marbled White-lipped Frog Auto Motif Mining

## Header

| Field | Value |
|---|---|
| Date | 2026-05-12 |
| Status | First run complete |
| BirdCLEF label | `22967` |
| Pipeline | `src/data_analysis/auto_motif/mining.py` (species-agnostic) |
| Band discovery | `src/data_analysis/auto_motif/band.py` |
| Output root | `artifacts/auto_motif/marbled_frog/` (older artifacts from before the layout change still at `artifacts/marbled_frog/motif_mining_auto/`) |
| Comparison script | `scripts/compare_auto_vs_existing.py` |

## What this is

A species-agnostic motif-mining pipeline that takes a species label and
discovers the frequency band, event size, and motif template directly from the
focal recordings — no per-species hand-tuning. Lives in `auto_motif/` (not
`marbled_frog/`) because Marbled Frog is just the first species we apply it
to.

Pipeline:

```text
wideband robust-z spectrograms (200-6000 Hz)
  → per-recording mean spectrum, MAD-normalized
  → cross-recording median consensus
  → contiguous high-z bands (>= median + k * MAD), top K
  → per band: hysteresis ROIs at in-band p99 / p90
  → per band: 2D KDE on (log dur, log bw); keep dominant mode within k_mad
  → per band: EM template refinement (median patch -> bimodal score split
    -> re-anchor at band-mean envelope argmax -> re-median)
  → per band: final detection with member-vs-background threshold
  → cross-family NMS + consolidation (reuses motif_template_calibration)
  → same 2-panel diagnostic plot as motif_recording_diagnostics
```

## Marbled Frog result

Discovered bands:

```text
band 0: 882-1712 Hz   (covers chip + upper part of primary)
band 1: 1710-2024 Hz  (covers secondary low)
band 2: 2034-2184 Hz  (covers secondary high)
```

Templates (see `plots/template_family_XX.png`):

- band 0: upward chirp 950 → 1600 Hz over ~0.3 s. This is the primary call's
  high-frequency portion.
- band 1: vertical bar 1750-2000 Hz, ~0.1 s. Secondary click low.
- band 2: vertical bar 2050-2175 Hz, ~0.1 s. Secondary click high.

EM converged in 2 iterations per band (`1 - cosine < 0.001`).

## A/B vs the manual pipeline

(`tables/ab_vs_existing.csv`, `tables/ab_curated_recovery_by_component.csv`)

| Metric | Auto | Manual |
|---|---|---|
| Consolidated events (all 8 recordings) | 227 | 154 |
| Median per-recording `|abs_diff|` | 8.5 | — |
| Auto precision vs manual | 0.53 | — |
| Auto recall vs manual | 0.84 | — |
| Overall curated recovery | 60.6% (20/33) | 100% (33/33) |

Per-component curated recovery:

```text
component             auto   manual
chip_1k_1p8k          5/5    5/5
secondary             4/4    4/4
secondary_independent 10/10  10/10
primary               1/4    4/4
primary_independent   0/10   10/10
```

Chips and secondaries: perfect recovery. Primary calls: 1/14.

## Why primary calls fail

Looking at `tables/auto_bands.csv` and the curated boxes, the primary call's
frequency support is 660-1296 Hz, but band 0 starts at 882 Hz. The consensus
spectrum dips below the `k_band_mad = 2.0` threshold around 880 Hz, so band 0
misses the lower 200 Hz of the primary's energy.

Band 0's template anchors on the upper part of the primary chirp (~1000-1600
Hz). When it scans the recording, it fires at the upper-frequency endpoint of
each primary call, but:

1. The score on actual primary calls is ~0.62, just under the
   member-vs-background threshold (~0.67), so band 0 detects sparsely.
2. When band 0 does detect, its peak time is offset 0.3-0.7 s from the curated
   peak (the chirp's upper endpoint vs. the curated centroid), so the recovery
   matcher fails the `peak_inside` check.
3. Bands 1 and 2 also fire on the primary call's harmonic content, but their
   support is 1710-2184 Hz, which has `freq_iou = 0` against the curated
   primary's 660-1296 Hz range. So `recovered = false`.

## Limitations of the band-per-template design

This recovery gap reveals the architectural limit: **one template per band**.
The manual pipeline gets 100% via 8 HDBSCAN clusters across all ROIs — multiple
templates per frequency band, each tuned to a different call sub-type. The
auto pipeline collapses everything in a band into a single template, which
fails when:

1. A band contains multiple temporally-distinct call types (band 0 has both
   primary and chip content; the KDE picks one mode, biasing the template).
2. The species' call extends below the band's lower edge.
3. The call's natural peak time differs from its highest-energy frame.

## Diagnostic plot

`plots/iNat1698105_band_envelope_vs_template_events.png` reproduces the same
2-panel layout as the existing
`artifacts/marbled_frog/motif_recording_diagnostics/plots/iNat1698105_band_envelope_vs_template_events.png`:
full spectrogram on top with template events (cluster-colored) and
band-envelope events (cyan = matched, red = unmatched), normalized envelope
panel below sharing x. The auto pipeline passes its primary discovered band
(882-1712 Hz) to the envelope detector and renders the full wideband
(200-6000 Hz) on the spectrogram y-axis.

## Possible next steps

In order of expected payoff:

1. **Multi-template per band**: cluster the EM-kept patches per band (k-means
   or HDBSCAN on the patch space) and emit one template per sub-cluster. This
   would recover the manual pipeline's per-cluster granularity without
   requiring human approval.
2. **Tail-extension of bands**: pad each discovered band downward until the
   consensus z falls below a *secondary* lower threshold (e.g. `k * 0.5`).
   This would extend band 0 from 882 Hz down to ~700 Hz, covering the primary
   call's full range.
3. **Anchor at chirp onset**: re-anchor M-step crops at the first frame where
   the band-mean envelope exceeds a fraction of its peak, not the argmax. This
   would align the template peak with what curated boxes call `peak_time_sec`.
4. **Per-recording band fallback**: if median consensus rejects a band that a
   *majority of recordings* would individually include, fall back to a less
   conservative consensus (e.g. Q40 across recordings).

The current pipeline is good enough as a no-human-input baseline for motifs
with relatively clean focal recordings — chips and secondaries are at
manual-quality recovery, and the diagnostic plots are immediately usable.
Primary-style multi-component chirps need one of the above extensions.

## Mitigations applied (2026-05-13)

All four mitigations from "Possible next steps" were implemented and
A/B-tested phase-by-phase. The big win was **onset anchoring** (#3) — the
other three were either neutral or counterproductive for Marbled Frog
specifically, but the infrastructure is in place for species where they
might matter.

### Per-component curated recovery by phase

| Phase | chip | primary | primary_indep | secondary | secondary_indep | overall | precision vs manual |
|---|---|---|---|---|---|---|---|
| Baseline | 5/5 | 1/4 | 0/10 | 4/4 | 10/10 | 20/33 (60.6%) | 0.53 |
| +tail-extension | 5/5 | 0/4 | 0/10 | 4/4 | 10/10 | 19/33 (57.6%) | 0.64 |
| +onset anchor | 4/5 | 3/4 | 5/10 | 4/4 | 9/10 | 25/33 (75.8%) | 0.58 |
| +multi-template (default off) | — | — | — | — | — | (same as +onset) | (same) |
| +per-recording fallback | 4/5 | 3/4 | 5/10 | 4/4 | 9/10 | 25/33 (75.8%) | 0.58 |

### What each phase did (or didn't do)

- **Tail-extension** (`auto_motif/band.py:discover_bands` hysteresis with
  `k_band_mad_lower = 1.0`): band 0 grew from 882–1712 Hz to 788–1774 Hz; band
  1 from 1710–2024 to 1648–2103; band 2 from 2034–2184 to 1960–2306. Precision
  went up because broader bands → bigger overlap → cross-family NMS suppresses
  duplicates. But the EM-derived `template_fmin_hz` for band 0 stayed at 906
  Hz (the actual energy is still in the upper chirp), so primary recovery
  dipped by 1. Net: a wash for Marbled Frog. Useful for species where the
  consensus dip is at the band edge.
- **Onset anchor** (`_refine_anchor_via_envelope` with
  `anchor_onset_fraction = 0.30`, propagated into `detect_with_auto_templates`):
  the M-step crop and the final-detection peak time both use the first frame
  whose band-mean envelope reaches 30% of its peak. This shifts peak times
  earlier by 0.1–0.3 s on ramped calls, landing inside curated primary boxes
  whose ground-truth peak is the chirp's leading edge. **Primary recovery
  jumped 0/4 → 3/4 and primary_independent 0/10 → 5/10.** Single biggest win.
- **Multi-template per band** (`em_loop` + `_subcluster_kept_patches` with
  PCA-then-HDBSCAN, default off): with `subcluster_pca_components = 8` and
  `subcluster_min_size = 6`, band 0 does split into 3 sub-clusters
  (sizes 17 / 6 / 34). But each sub-template's member-vs-background threshold
  is *higher* than the parent's (tight sub-clusters → high Q25 member scores)
  so each sub-template fires sparsely. Net: recovery tanked to ~23% when
  enabled. Defaulted off; the path is wired end-to-end and can be flipped on
  per-species if a band is known multi-modal and threshold derivation is
  adjusted (e.g. share the parent threshold across sub-templates).
- **Per-recording band fallback** (run `discover_bands` on each per-recording
  spectrum and promote candidate bands agreed on by ≥ N recordings that the
  consensus rejected): never fired on Marbled Frog — all 3 bands were already
  found via the median consensus. The fallback path has since been removed
  to keep band discovery to a single consensus pass; reintroduce if a future
  species has a sparse component visible in only a minority of recordings.

### Remaining gap

Still 1/5 chip, 1/4 primary, 5/10 primary_indep, 1/10 secondary_indep below
manual. The remaining primary calls fall just under the
member-vs-background threshold (~0.65) at scores ~0.60. Two paths to close
the rest:

1. **Drop the threshold floor** from 0.45 toward 0.40 and lean on the
   final-detection NMS to filter noise instead of the threshold. Risk:
   precision drop.
2. **Frequency-aware curated matching**: the recovery test requires
   `freq_iou >= 0.10` against the curated box; for primary calls whose
   harmonic energy is in bands 1 and 2 (1648–2306 Hz) but whose curated
   fmin/fmax is 660–1296 Hz, freq_iou is artificially low. A harmonic-aware
   matcher would credit these detections. This is an evaluator change, not
   a detector change.
