# Auto Motif Mining Pipeline

| Field | Value |
|---|---|
| Date | 2026-05-15 |
| Status | Species-agnostic reference; stable. Current Marbled Frog recovery in [Results](#results). |
| Pipeline code | `src/data_analysis/auto_motif/mining.py` |
| Band discovery | `src/data_analysis/auto_motif/band.py` |
| Shared evaluators | `src/data_analysis/common/curated_eval.py` |
| Shared plot | `src/data_analysis/common/recording_plot.py` |
| Comparison harness | `scripts/compare_auto_vs_existing.py` |
| Per-species output | `artifacts/auto_motif/<species_dir>/` |

## Contents

- [Goal](#goal) · [Inputs](#inputs)
- [Library choices](#library-choices) · [Pipeline stages](#pipeline-stages)
- [Configuration reference](#configuration-reference) · [CLI](#cli) · [Output schema](#output-schema)
- [Toggles and troubleshooting](#toggles-and-troubleshooting)
- [Reuse from manual pipeline](#reuse-from-manual-pipeline)
- [Results](#results) · [Limitations](#limitations-and-follow-ups)

## Goal

Given a BirdCLEF species label, discover the species' frequency band(s),
event size, and motif template from focal recordings *with no per-species
hand-tuning*. The pipeline is intended as a no-human-input baseline for
species where most calls are reasonably clean — chips, secondaries, and
short tonal calls in particular. Multi-component chirps (primary calls
with ramps) need the onset-anchor mitigation enabled (default).

## Inputs

- BirdCLEF label (`str`, e.g. `"22967"`).
- Focal training audio at `train_audio/<label>/*.ogg` (resolved via
  `common.paths.recording_files_for_label`).
- Optional: curated motif boxes CSV for recovery evaluation
  (schema: `filename, start_sec, end_sec, fmin_hz, fmax_hz, component, …`).
  Not required; if absent the curated-recovery stage is skipped.

## Library choices

Each library dependency is justified independently. Aggregated, the
2026-05-13 swap delivered +9 pp Marbled Frog recovery; commit history has
the before/after narrative.

| Tool | Used for | Why |
|---|---|---|
| `librosa.pcen` | Per-bin time-frequency normalization | AGC + DRC produces a perceptually-flat representation that tracks transients better than dB-MAD. |
| `maad.features.shape_features` + `centroid_features` | 26-D Gabor ROI descriptors + intensity centroid | Discriminate call morphology directly. A prior `(log dur, log bw)` KDE conflated chirps with tonal notes of similar size. |
| `sklearn.cluster.HDBSCAN` | Multi-template emission per band | One cluster decision per band, made on a tight feature space before EM, instead of two parallel paths (KDE pre-EM + patch sub-clustering post-EM). |
| `opensoundscape.ribbit` | Band-level periodicity score (opt-in) | Frog/insect motifs have stable inter-call intervals; runs only when `ribbit_score_min` is set, dropping bands below the threshold. |
| `maad.features.bioacoustics_index` | Per-recording activity score (opt-in pre-rank) | Cheap amplitude-STFT recording filter; runs only when `min_activity_quantile` is set. |

What is deliberately hand-rolled (no clean library equivalent surfaced):

- **Band discovery** — a *robust core + recall extension* (`band.py`): the
  per-recording MAD-normalized **median** consensus + hysteresis gives the
  high-precision dominant bands; an **exemplar-weighted** consensus then
  recovers the sparse / frequency-jittered bands the median washes out.
  This is the literature's robust-estimator-core + recall-extension pattern
  (a Long-Term-Average-Spectrum variant under a multiple-instance framing,
  with confidence/exemplar weighting); no single off-the-shelf detector
  produces "frequency bands where this species' energy recurs across
  recordings", so the composition is hand-rolled. No tuned constant — the
  exemplar weight is the species' own dominant-band strength.
- **Envelope-onset peak anchoring** — depends on band-mean envelope shape;
  see `_refine_anchor_via_envelope`.
- **EM score thresholding** — fixed quantile (default p95) of the score
  distribution, clipped to `[threshold_floor, threshold_ceiling]`. Prior
  bimodal-split machinery (KDE valley check, GMM-2 crossover, Otsu) never
  fired on real data and was removed.
- **NMS + cross-family consolidation** — reuses
  `marbled_frog.motif_template_calibration` helpers (see [Reuse](#reuse-from-manual-pipeline)).

## Pipeline stages

```text
[0]  (opt-in) recording activity index (maad.features.bioacoustics_index over WIDEBAND)
[1]  → wideband PCEN spectrograms (librosa.pcen, 200-6000 Hz)
[2]  → per-recording time-p95 spectrum, MAD-normalized
       → robust CORE: median consensus → hysteresis bands (top core_bands)
       → RECALL EXTENSION: exemplar-weighted consensus (weight = each
         recording's strength in the core's anchor band) → hysteresis
         bands, merged, kept only where the core had no coverage
       → core + extension, capped at max_bands
[3]  → (opt-in) per band: opensoundscape.ribbit periodicity score
                drops bands below `ribbit_score_min`
[4]  → per band: maad hysteresis ROIs at in-band p99 (seed) / p90 (grow)
[5]  → per band: maad.features.shape_features (26-D Gabor) +
                 centroid_features → HDBSCAN → one family per non-noise cluster
[6]  → per (band, cluster): EM template refinement (median patch → p95
                 quantile score split → re-anchor at band-mean envelope ONSET
                 → re-median; stop on cosine)
[7]  → per family: member-vs-background threshold; peak-pick; NMS;
                  cross-family consolidation (reused from marbled_frog
                  motif_template_calibration)
[8]  → curated recovery + 2-panel diagnostic plot per recording
[8b] → per-motif-family activation diagnostic (detection boxes + score
       track across all recordings, one figure per template)
```

### 0. Recording pre-ranking (opt-in)

When `AutoConfig.min_activity_quantile` is set,
`rank_recordings_by_activity(files, flim=WIDEBAND)` computes
`maad.features.bioacoustics_index` per focal recording on an amplitude STFT
over `WIDEBAND`, writes `tables/recording_activity.csv`, and drops
recordings below the quantile before `load_focal_z` materializes their
spectrograms. Otherwise this stage is skipped entirely (no compute, no
table).

### 1. Wideband spectrograms

`load_focal_z(files, band=WIDEBAND)` caches
`(db_band, s_band, freqs_band, times)` per recording via
`common/spectrogram.wideband_pcen_spectrogram`. All downstream stages
(percentile-based hysteresis, patch-normalized NCC, envelope-relative onset
anchor, Q25/Q99.5 thresholds) are scale-invariant, so PCEN flows through
the same NMS/consolidation pipeline that previously consumed robust-z. The
cache variable name `z` is preserved despite holding PCEN values.

### 2. Band discovery

`auto_discover_bands(cache, config=BandDiscoveryConfig())` in `mining.py`,
delegating to `discover_bands` in `band.py`.

For each recording, the per-frequency-bin time-p95 of PCEN is normalized by
its own MAD-sigma. (Time-mean was used with robust-z but PCEN's AGC
flattens long-running averages — the high time-quantile preserves bins with
concentrated transient activity.)

**Robust core.** The *median* across recordings forms the consensus
spectrum. Contiguous runs above `med + k_band_mad * sigma` seed each band;
each seed grows outward (hysteresis) while consensus exceeds
`med + k_band_mad_lower * sigma`, clamped at the adjacent run so neighbors
do not merge. The top `core_bands` runs by peak score are the high-precision
core (the dominant chip/primary complex). The median is robust to a single
recording dominated by a heterospecific, but it *washes out* a real motif
whose energy peak jitters in frequency across recordings: at any single bin
that motif is a minority, so the per-bin median drops it.

**Recall extension.** Each recording is weighted by its own peak salience in
the core's top (anchor) band — the species' undisputed dominant signal, so
clean exemplars define the subtler bands and marginal/noisy recordings are
downweighted (no tuned constant; `exemplar_weights`). Band discovery is
re-run on this `weighted_consensus`; the resulting runs are merged
(`merge_overlapping_bands`) and only those that overlap *no* core band are
kept (`bands_outside`) — genuinely new frequency territory. Core +
extension, capped at `max_bands`, is returned. This is a robust-core +
recall-extension design (median for precision, exemplar weighting for
recall); see [Library choices](#library-choices).

### 3. RIBBIT periodicity scoring (opt-in)

When `AutoConfig.ribbit_score_min` is set, `score_band_periodicity(files,
bands, ...)` runs `opensoundscape.ribbit.ribbit` on each (recording, band)
pair across overlapping clips of `ribbit_clip_duration_sec`. The per-clip
RIBBIT score (power inside `ribbit_pulse_rate_hz` Hz of the band-envelope
spectrum) is aggregated to `(max, p95, median)` per band, appended to
`auto_bands.csv`, and used to drop bands below the threshold. Otherwise
this stage is skipped entirely.

### 4. Loose ROI proposal

`propose_loose_rois(cache, bands, *, config)`. For each band, scikit-maad
`create_mask` runs hysteresis ROI detection on the in-band PCEN
spectrogram using percentile thresholds `seed_percentile` and
`grow_percentile` *computed per recording* (so absolute value levels don't
have to be calibrated across species). `select_rois` extracts bounding
boxes; loose duration/bandwidth filters keep biological-scale events.

### 5. Shape-feature clustering

`shape_cluster_refine(loose_rois, cache, *, min_cluster_size)`. Per band:
26-D Gabor shape descriptors from `maad.features.shape_features` (needs
`util.format_features` for pixel-coord columns) + intensity centroid from
`centroid_features`. Z-standardize the combined matrix; cluster with
`HDBSCAN(min_cluster_size=shape_cluster_min_size, allow_single_cluster=True)`.

Each non-noise cluster becomes a candidate motif family flowing into EM as
`motif_cluster = band_id * 100 + shape_cluster_id`; noise (label `-1`) is
dropped. `tables/shape_features.csv` carries one row per loose ROI with all
shape + centroid columns and the assigned cluster (or `-1` for noise).

### 6. EM template refinement

`em_loop(bands, cache, refined_rois, *, config)` runs one EM pass per
`(band_id, shape_cluster_id)` partition; `em_step` runs one E + M pass.

- **Init**: `init_template` takes the median patch of cluster-member ROIs
  anchored at each ROI's peak time. Support (`template_pre_sec`,
  `template_post_sec`, `template_fmin_hz`, `template_fmax_hz`) comes from
  `canonical_support` on the surviving ROIs, with `freq_clip` set to the
  discovered band.
- **E-step**: cross-correlate the template against the in-band PCEN
  spectrogram across every recording (`_score_track`). Peak-pick at
  `quantile_threshold(scores, quantile=em_threshold_quantile)`, clipped to
  `[threshold_floor, threshold_ceiling]`, with peak distance
  `max(0.12, 0.60 * template_duration_sec)`.
- **M-step**: each kept peak is re-anchored at the *first* frame whose
  band-mean envelope exceeds `anchor_onset_fraction * max(envelope)` inside
  the support window (`_refine_anchor_via_envelope`). Re-median the patches
  to form the new template.
- Stop when `1 − cosine(new, prev) < template_change_eps` after at least one
  iteration, or after `max_em_iter`. Drop families with `< min_repetitions`
  kept peaks.

`em_trace.csv` records per-iteration `band_id, shape_cluster_id, n_kept,
threshold, method, cosine_to_prev`. With the bimodal-split machinery
removed, `method` is always `quantile` (or `empty` for degenerate cases).

### 7. Final detection + NMS + consolidation

`detect_with_auto_templates(families, cache, *, config)` re-runs the
template against every recording and re-derives the threshold from
`min(Q25 members, Q99.5 background)` clipped to
`[threshold_floor, threshold_ceiling]` (mirrors
`marbled_frog.motif_template_calibration.estimate_thresholds`). Peaks are
re-anchored via the same envelope-onset helper. Then:

- `nms_events` (from `marbled_frog.motif_template_calibration`) suppresses
  within-family overlaps at `nms_time_iou` IoU and
  `nms_peak_distance_sec` peak spacing.
- `consolidate_events` merges cross-family detections with overlapping time
  (`consolidate_min_time_iou` IoU and `consolidate_peak_distance_sec` peak
  proximity) into one biological event box, recording
  `motif_clusters_present` for provenance.

### 8. Curated recovery + diagnostics

If a curated CSV is provided, `common.curated_eval.curated_recovery`
matches consolidated events against curated boxes using time-IoU + freq-IoU +
peak-inside criteria. `detection_stage_metrics` reports recovery rate at
raw / NMS / consolidated stages.

`common.recording_plot.plot_recording_diagnostic` produces a 2-panel plot
(wideband spectrogram with cluster-colored event boxes + envelope panel
sharing x) per recording, matching the manual pipeline's layout. The
envelope panel uses the discovered band with the most refined ROIs so the
trace reflects the densest call activity instead of the highest-peak band.

### 8b. Per-motif activation diagnostic

`common.recording_plot.plot_motif_activation_diagnostic` produces one
figure per template family (`plots/motif_activation_XXXX.png`). The top
panel is a **label-free template-quality screen**: histograms of the
template's EM-member score-track values vs background values, with the
detection threshold and the **member-vs-background separation** (`d'` and
ROC-`AUC`). Below it, each focal recording is a two-panel block: the
spectrogram zoomed to the family's template band with a green box around
every detected activation (the per-family NMS events), and the dense
template score track with the detection threshold (red dashed) and
detected peaks marked. Score y-range is shared across recordings for
comparability; recordings where the family never fires are still shown
(flat track) so misses / band-vs-call mismatches are visible per template.

The separation (`_score_separation` in `mining.py`) is computed from the
member/background arrays already collected to derive the detection
threshold, and persisted to `auto_templates_metadata.csv`
(`member_bg_dprime`, `member_bg_auc`, `n_member_scores`,
`n_background_scores`). **Necessary-not-sufficient:** members are the
template's own EM picks (optimistically biased) and background is
contaminated by any real calls EM missed (deflates the score). A *low*
value definitively flags a non-discriminative template; a *high* value
proves internal self-consistency, not that the picks are real calls.
Complement it with score margin, cross-template corroboration
(`consolidated_events.n_component_detections`), periodicity, and human
spot-check.

## Configuration reference

### `BandDiscoveryConfig` (`auto_motif/band.py`)

| Knob | Default | Rationale |
|---|---|---|
| `k_band_mad` | `2.0` | Primary threshold: `med + 2*MAD` ≈ p98 of consensus |
| `k_band_mad_lower` | `1.0` | Secondary (hysteresis) threshold for grow phase |
| `min_band_width_hz` | `150.0` | ~10 FFT bins at sr=32k, nfft=2048 |
| `core_bands` | `3` | Robust median-consensus core size (the conventional high-precision setting; the chip/primary complex) |
| `max_bands` | `5` | Core + exemplar-weighted recall-extension bands, total |
| `band_pad_hz` | `40.0` | One FFT bin each side |

### `AutoConfig` (`auto_motif/mining.py`)

| Knob | Default | Rationale |
|---|---|---|
| `band` | `BandDiscoveryConfig()` | Nested band-discovery config |
| `seed_percentile` | `0.99` | In-band PCEN p99 hysteresis seed |
| `grow_percentile` | `0.90` | In-band PCEN p90 hysteresis grow |
| `loose_min_duration_sec` | `0.02` | Loose; shape clustering narrows |
| `loose_max_duration_sec` | `2.0` | Loose; shape clustering narrows |
| `loose_min_bandwidth_hz` | `50.0` | Loose; shape clustering narrows |
| `loose_max_bandwidth_hz` | `3000.0` | Loose; shape clustering narrows |
| `min_area_pixels` | `6` | Matches `select_rois` default |
| `closing_freq_bins` | `1` | Morphological closing for ROI gap-filling |
| `closing_time_bins` | `3` | Morphological closing for ROI gap-filling |
| `shape_cluster_min_size` | `5` | HDBSCAN `min_cluster_size` on shape features |
| `max_em_iter` | `5` | EM usually converges in 2-3 |
| `template_change_eps` | `0.01` | Cosine > 0.99 |
| `min_repetitions` | `5` | Drop families below this kept-peak count |
| `nms_time_iou` | `0.35` | Same as manual pipeline |
| `nms_peak_distance_sec` | `0.12` | Same as manual pipeline |
| `consolidate_peak_distance_sec` | `0.25` | Same as manual pipeline |
| `consolidate_min_time_iou` | `0.05` | Same as manual pipeline |
| `threshold_floor` | `0.45` | Lower clip on member-vs-bg threshold |
| `threshold_ceiling` | `0.90` | Upper clip on member-vs-bg threshold |
| `em_threshold_quantile` | `0.95` | Quantile of EM score distribution used as the kept-peak threshold |
| `anchor_onset_fraction` | `0.30` | First envelope frame ≥ 30% of peak; aligns peak_time with curated leading-edge |
| `ribbit_pulse_rate_hz` | `(0.3, 30.0)` | Pulse-rate window for `opensoundscape.ribbit` (only used when `ribbit_score_min` is set) |
| `ribbit_clip_duration_sec` | `10.0` | RIBBIT analysis clip length per recording (only used when `ribbit_score_min` is set) |
| `ribbit_score_min` | `None` | Opt-in band filter; setting it enables RIBBIT scoring + drop |
| `min_activity_quantile` | `None` | Opt-in recording filter; setting it enables bioacoustics_index pre-rank + drop |

## CLI

```bash
python -m data_analysis.auto_motif.mining \
  --label 22967 \
  --curated artifacts/marbled_frog/motif_boxes/curated/tables/curated_motif_boxes.csv \
  --output-dir artifacts/auto_motif/marbled_frog  # optional; defaults via LABEL_TO_DIR
```

`LABEL_TO_DIR` (in `mining.py`) maps known labels to a species directory
name. Unknown labels fall back to `artifacts/auto_motif/<label>/`.

## Output schema

`artifacts/auto_motif/<species_dir>/`:

| Path | Schema / contents |
|---|---|
| `summary.md` | Auto-generated run summary; lists tables/plots/templates with discovered bands. |
| `tables/recording_activity.csv` | `filename, bioacoustics_index` (only when `min_activity_quantile` is set). |
| `tables/auto_bands.csv` | `band_id, fmin_hz, fmax_hz, bandwidth_hz` plus `ribbit_score_max, ribbit_score_p95, ribbit_score_median, n_clips_scored` when `ribbit_score_min` is set. |
| `tables/auto_band_consensus.csv` | `filename, freq_hz, z_normalized`; `filename == "__consensus__"` is the consensus that surfaced the extension bands (exemplar-weighted when a core was found, else median). |
| `tables/loose_rois.csv` | Per-band hysteresis ROIs (loose dur/bw filter). |
| `tables/shape_features.csv` | Per-ROI 26-D Gabor shape + centroid features and HDBSCAN cluster (-1 = noise). |
| `tables/em_trace.csv` | `band_id, shape_cluster_id, iter, n_kept, threshold, method, cosine_to_prev`. |
| `tables/auto_templates_metadata.csv` | One row per accepted family: band/shape-cluster IDs, threshold (EM), support, `member_bg_dprime`, `member_bg_auc`, `n_member_scores`, `n_background_scores`. |
| `tables/template_score_tracks.csv` | Dense (filename, motif_cluster, time, score). |
| `tables/template_match_events_raw.csv` | Raw peak-pick detections per family. |
| `tables/template_match_events_nms.csv` | Within-family NMS survivors. |
| `tables/consolidated_events.csv` | Cross-family consolidated events. Schema matches manual pipeline. |
| `tables/event_summary.csv` | Per-(filename, cluster) event counts. |
| `tables/inter_event_intervals.csv` | Consecutive-event intervals per recording / cluster. |
| `tables/curated_recovery.csv` | Best detection per curated box + IoU/peak-inside flags. |
| `tables/detection_stage_metrics.csv` | Raw vs NMS vs consolidated reduction + recovery. |
| `plots/auto_band_consensus.png` | Per-recording + median consensus spectrum with discovered bands. |
| `plots/shape_clusters.png` | Per-band SVD-projected scatter of shape features colored by HDBSCAN cluster. |
| `plots/em_convergence.png` | Per-family `1 - cosine` vs iter. |
| `plots/template_family_XXXX.png` | Final template per `motif_cluster` (band*100 + shape_cluster). |
| `plots/motif_activation_XXXX.png` | Per-family activation diagnostic: detection boxes + score track across all recordings. |
| `plots/{stem}_band_envelope_vs_template_events.png` | 2-panel diagnostic per recording. |
| `templates/family_XXXX_template.npz` | Fixed-support template arrays + support dict. |

## Toggles and troubleshooting

### When to flip a knob

- **`shape_cluster_min_size`** controls how aggressively HDBSCAN splits ROIs
  within a band. Raise (e.g. `8–12`) when bands are noisy and you want only
  large, coherent clusters surviving. Lower (e.g. `3–4`) for species with
  small but distinct call modes.
- **`ribbit_score_min`** — set a numeric threshold (e.g. `0.05`) to enable
  RIBBIT band-level scoring and drop bands below it. Leave at `None` to
  skip the stage entirely (saves a per-(file,band) OpenSoundscape pass).
- **`min_activity_quantile`** — set (e.g. `0.25`) to enable
  bioacoustics_index pre-ranking and drop the bottom 25% of recordings.
  Leave at `None` to skip the stage entirely.
- **`anchor_onset_fraction` (default `0.30`)**: lower (e.g. `0.10–0.20`) for
  species where curated peak_time is the *very* leading edge of a call;
  higher (e.g. `0.40–0.50`) for impulsive calls where onset ≈ argmax.
- **`threshold_floor`**: drop to `0.40` if recall is the bottleneck and the
  remaining curated boxes score in the 0.55–0.65 range. Expect a precision
  hit.
- **`em_threshold_quantile`**: lower (e.g. `0.90`) to keep more EM-stage
  candidate peaks per iteration; raise (e.g. `0.98`) for stricter
  template-member selection. The clipped value is bounded by
  `[threshold_floor, threshold_ceiling]`.
- **`band.k_band_mad_lower`**: tune the secondary hysteresis threshold. The
  default `1.0` extends bands by ~1 FFT bin until consensus falls below
  `med + 1*MAD`. Lower it (e.g. `0.5`) to extend further; raise (e.g. `1.5`)
  to keep bands tight.
- **`band.core_bands`** (default `3`): the conservative median-consensus
  core size. Raise only if a species' high-precision dominant structure has
  more than three components; the recall extension already covers the rest.
- **`band.max_bands`** (default `5`): total bands after the
  exemplar-weighted recall extension. Raise for species with many distinct
  call components, but watch within-band consolidation artifacts when
  extension bands overlap the core.

### Debugging a new species (symptom → artifact)

- **No bands discovered** → `tables/auto_band_consensus.csv`, plot
  `auto_band_consensus.png`. Lower `band.k_band_mad` or widen `WIDEBAND`.
- **Few/empty loose ROIs in a real band** → `tables/loose_rois.csv`. Lower
  `seed_percentile` / `grow_percentile` or relax loose dur/bw bounds.
- **HDBSCAN returns all noise** → `tables/shape_features.csv`, plot
  `shape_clusters.png`. Lower `shape_cluster_min_size`.
- **EM drops the family** → `tables/em_trace.csv`. Look for `method =
  skipped_too_few_rois` or `dropped_below_min_repetitions`; consider
  lowering `min_repetitions` or `threshold_floor`.
- **Detections look spurious / "motif isn't good"** → `member_bg_dprime` /
  `member_bg_auc` in `auto_templates_metadata.csv` and the top panel of
  `plots/motif_activation_XXXX.png`. Low values (overlapping member/
  background histograms) = non-discriminative template. Cross-check the
  per-detection score margin and `consolidated_events.n_component_detections`
  (corroboration across bands).
- **Low curated recovery despite detections** → check
  `tables/detection_stage_metrics.csv` for raw → NMS → consolidated drop.
  Cross-family consolidation can absorb adjacent secondaries (see
  [Limitations](#limitations-and-follow-ups)).

## Reuse from manual pipeline

The auto pipeline imports and reuses (does not duplicate) these helpers
from `marbled_frog/`, all of which are species-agnostic in behavior:

- `motif_bank.canonical_support(rois, *, freq_clip)` — patch shape from ROIs.
- `motif_bank.fixed_support_crop(z, freqs, times, peak_time_sec, support)` —
  fixed-shape patch extraction.
- `motif_template_calibration.normalize_resize_patch` — patch normalization.
- `motif_template_calibration.scan_cluster` — score-track computation.
- `motif_template_calibration.nms_events` — within-family NMS.
- `motif_template_calibration.consolidate_events` — cross-family consolidation.
- `motif_template_calibration.summarize_events` — per-family event summaries.
- `motif_recording_diagnostics.band_envelope_peaks` — envelope panel of the
  diagnostic plot.

These helpers live under `marbled_frog/` for historical reasons; their
implementations have no Marbled Frog-specific logic. See
[Limitations](#limitations-and-follow-ups) on lifting them into `common/`
once a second species drives the need.

## Results

### Marbled Frog (label 22967)

After the 2026-05-15 robust-core + exemplar-weighted recall extension to
band discovery: **~0.96 mean per-component curated recovery**, up from the
~0.78 median-only baseline. Per-component (representative run; downstream
HDBSCAN/EM give minor run-to-run variance, band discovery is
deterministic):

| Component | Curated | Recovered | Rate | Manual |
|---|---|---|---|---|
| chip_1k_1p8k | 5 | 4 | 0.80 | 1.00 |
| primary | 4 | 4 | 1.00 | 1.00 |
| primary_independent | 10 | 10 | 1.00 | 1.00 |
| secondary | 4 | 4 | 1.00 | 1.00 |
| secondary_independent | 10 | 10 | 1.00 | 1.00 |

Discovered bands: 3 median-core chip/primary bands `(1351–1602)
(1132–1352) (866–1212)` Hz + 2 exemplar-weighted secondary bands
`(1839–1989) (2382–2743)` Hz (the secondary's low and high lobes). The
prior diagnosis that `secondary = 0/4` was a consolidation-policy artifact
was **wrong**: it was a band-discovery failure — the cross-recording
*median* washed out the secondary because its energy peak jitters in
frequency between recordings. Exemplar-weighted recall extension recovers
it (0/4 → 4/4, 9/10 → 10/10) while the median core preserves the PCEN
primary gain (precision unchanged at ~0.61 on *fewer* events). The only
remaining gap vs. manual is one chip box. RIBBIT scores were uniformly low
(~0.08–0.14) for Marbled Frog (no regular pulse-rate signature), so the
score is reported but not used as a filter.

## Limitations and follow-ups

- **Reported recovery is recall, not precision.** The curated boxes are 33
  hand-picked exemplars in only 3 of 8 focal recordings — not an
  exhaustive annotation. `curated recovery ≈ 0.96` means "of the selected
  example calls, ~96% were recovered"; it says nothing about how many of
  the ~200+ detections per family are spurious. There is no labeled
  not-a-call set, so true precision is **unmeasured**. The
  `member_bg_dprime`/`auc` separation is a label-free template-quality
  *screen* (necessary-not-sufficient, see [8b](#8b-per-motif-activation-diagnostic)),
  not a precision measure. A small *exhaustive* annotation of 1–2
  recordings is the only way to get a true precision anchor.
- **Frequency-IoU strictness in recovery eval.** Bands that fire on a
  call's harmonic content but don't overlap the curated fundamental are
  scored as misses. Consider a harmonic-aware matcher in
  `common/curated_eval.py`.
- **Exemplar weight = anchor-band strength.** A clean primary-call
  exemplar is assumed to also express the subtler components. True for
  Marbled Frog (the high-weight recordings carry strong secondary energy
  too) but not guaranteed in general; a species whose secondary appears
  only in recordings *weak* in the primary would be downweighted. SNR- or
  classifier-confidence-based exemplar weights are the fallback if a future
  species breaks this assumption.
- **Single anchor band.** The recall extension weights by the *one*
  top-ranked core band. Fine when the dominant call is unambiguous;
  revisit (e.g. multi-anchor weighting) if a species has two co-dominant
  call types in disjoint bands.
- **Cache memory.** `load_focal_z` holds wideband spectrograms for every
  recording in RAM. For species with many long recordings, lazy iteration
  + on-disk caches may be needed.
- **Helper provenance.** `canonical_support`, `fixed_support_crop`,
  `normalize_resize_patch`, `nms_events`, `consolidate_events`,
  `summarize_events`, `band_envelope_peaks` live under `marbled_frog/`
  but are species-agnostic. Lift into `common/motif_template.py` once a
  second species drives the need — don't pre-emptively touch the manual
  pipeline.
