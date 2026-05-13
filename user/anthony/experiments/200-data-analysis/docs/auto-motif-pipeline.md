# Auto Motif Mining Pipeline

## Header

| Field | Value |
|---|---|
| Date | 2026-05-13 |
| Status | Species-agnostic reference; Marbled Frog 28/33 (84.8%) consolidated curated recovery after PCEN + library-tool migration. |
| Pipeline code | `src/data_analysis/auto_motif/mining.py` |
| Band discovery | `src/data_analysis/auto_motif/band.py` |
| Shared evaluators | `src/data_analysis/common/curated_eval.py` |
| Shared plot | `src/data_analysis/common/recording_plot.py` |
| Comparison harness | `scripts/compare_auto_vs_existing.py` |
| Per-species output | `artifacts/<species_dir>/motif_mining_auto/` |

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

## Design rationale

The 2026-05-13 migration leans on four library tools instead of hand-rolled
equivalents. Each replacement is justified independently and aggregated
gave +9 pp Marbled Frog recovery (25/33 → 28/33).

| Concern | Old (hand-rolled) | New (library) | Why |
|---|---|---|---|
| Per-bin time-frequency normalization | Robust-z (median + MAD + Gaussian smoothing, clipped) | `librosa.pcen` | PCEN's AGC + DRC is the well-studied perceptual normalizer; one library call replaces ~30 lines and tracks transients better than dB-MAD. |
| ROI shape characterization | 2-D KDE on `(log dur, log bw)` keeping the dominant mode | `maad.features.shape_features` (26-D Gabor) + `centroid_features` → HDBSCAN | The KDE conflated chirps with tonal notes of similar size. 26-D shape descriptors discriminate call morphology directly. |
| Multi-template emission per band | Post-EM PCA → HDBSCAN on patches (documented regression on Marbled Frog) | Same HDBSCAN as above, *before* EM, on shape-feature space | One source of multi-template emission instead of two parallel paths; cluster decision happens on a tighter, semantically richer feature space. |
| Band-level periodicity prior | None | `opensoundscape.ribbit` per (recording, band), scores aggregated to `max/p95/median` | Frog/insect motifs have stable inter-call intervals; this gives a numeric band quality signal that can later filter or weight. Defaults to score-only. |
| Recording quality pre-screen | None | `maad.features.bioacoustics_index` per file | Cheap, computed from amplitude STFT (no full spectrogram cache). Defaults to score-only; opt-in filter via `min_activity_quantile`. |

Two adaptation notes followed from the PCEN swap:

- **Band discovery** (`band.py`) switched from per-bin time-mean to time-p95
  before MAD normalization. PCEN's AGC flattens long-run averages so the
  mean is nearly constant across bins; a high time-quantile preserves bins
  with concentrated transient activity (target calls).
- **All downstream stages are scale-invariant**: percentile-based hysteresis,
  patch normalization (zero-mean unit-std before NCC), envelope-relative
  onset anchoring, and Q25/Q99.5 member-vs-background thresholds. So PCEN
  and robust-z values flow through the same NMS/consolidation pipeline
  unchanged. The cache variable name `z` is preserved despite holding PCEN
  values to keep the diff small.

What we explicitly *kept* hand-rolled:

- **Band discovery** itself (per-recording MAD-normalized consensus +
  hysteresis): no library equivalent surfaces the "frequency bands where
  this species' energy is consistently concentrated".
- **NMS + cross-family consolidation**: no library equivalent; the
  `marbled_frog.motif_template_calibration` helpers are reused as-is.
- **Envelope-onset peak anchoring**: a +5-box mitigation that depends on
  band-mean envelope shape; no clean library substitute.
- **Bimodal score-distribution split** (GMM-2 crossover → Otsu →
  quantile fallback): works for thresholding kept-vs-rejected EM peaks;
  no library equivalent.

## Pipeline stages

```text
[0]  recording activity index (maad.features.bioacoustics_index over WIDEBAND)
[1]  → wideband PCEN spectrograms (librosa.pcen, 200-6000 Hz)
[2]  → per-recording time-p95 spectrum, MAD-normalized
       → cross-recording median consensus
       → contiguous bands via hysteresis (primary k*MAD seed,
                                          secondary k_lower*MAD grow)
       → optional per-recording fallback for sparse minority bands
[3]  → per band: opensoundscape.ribbit periodicity score (max/p95/median over clips)
[4]  → per band: maad hysteresis ROIs at in-band p99 (seed) / p90 (grow)
[5]  → per band: maad.features.shape_features (26-D Gabor) +
                 centroid_features → HDBSCAN → one family per non-noise cluster
[6]  → per (band, cluster): EM template refinement (median patch → bimodal
                 score split → re-anchor at band-mean envelope ONSET → re-median;
                 stop on cosine)
[7]  → per family: member-vs-background threshold; peak-pick; NMS;
                  cross-family consolidation (reused from marbled_frog
                  motif_template_calibration)
[8]  → curated recovery + 2-panel diagnostic plots
```

### 0. Recording pre-ranking

`rank_recordings_by_activity(files, flim=WIDEBAND)` computes
`maad.features.bioacoustics_index` per focal recording on an amplitude STFT
over `WIDEBAND`. Writes `tables/recording_activity.csv` and (when
`AutoConfig.min_activity_quantile` is set) drops recordings below the
quantile before `load_focal_z` materializes their spectrograms.

### 1. Wideband spectrograms

`load_focal_z(files, band=WIDEBAND)` in `mining.py` calls
`wideband_pcen_spectrogram` from `common/spectrogram.py` — per-channel energy
normalization (PCEN; Wang et al. 2017) via `librosa.pcen`. The AGC + dynamic-
range compression produces a perceptually-flat representation whose downstream
consumers (percentile-based hysteresis, patch-normalized NCC, envelope-
relative onset anchor) are all scale-invariant. The cache holds
`(db_band, s_band, freqs_band, times)` per recording; ``s_band`` is the PCEN
spectrogram (still referred to in code as ``z`` for historical reasons).

### 2. Band discovery

`auto_discover_bands(cache, config=BandDiscoveryConfig())` in `mining.py`,
delegating to `discover_bands` in `band.py`.

For each recording, the per-frequency-bin time-p95 of PCEN is normalized by
its own MAD-sigma. (Time-mean was used with robust-z but PCEN's AGC
flattens long-running averages — the high time-quantile preserves bins with
concentrated transient activity.) The median across recordings forms the
consensus spectrum. Contiguous runs above `med + k_band_mad * sigma` seed
each band; each seed then grows outward (hysteresis) while consensus
exceeds `med + k_band_mad_lower * sigma`, clamped at the adjacent run so
neighbors do not merge.

After the consensus pass, the same procedure is applied independently to
each per-recording spectrum. Candidate bands that ≥ `fallback_min_recordings`
recordings agree on (≥ `fallback_overlap_fraction` IoU) but consensus
missed are promoted, up to `max_bands`. `auto_bands.csv` tags each band
with `discovery_method ∈ {"consensus", "fallback_per_recording"}`.

### 3. RIBBIT periodicity scoring

`score_band_periodicity(files, bands, ...)` runs
`opensoundscape.ribbit.ribbit` on each (recording, band) pair across
overlapping clips of `ribbit_clip_duration_sec`. The per-clip RIBBIT score
(power inside `ribbit_pulse_rate_hz` Hz of the band-envelope spectrum)
is aggregated to `(max, p95, median)` per band and appended to
`auto_bands.csv`. Bands carrying repetitive call structure score high;
incidental noise bands score low.

When `AutoConfig.ribbit_score_min` is set, bands below the threshold are
dropped before downstream stages. Default `None` = score-only, no drop.

### 4. Loose ROI proposal

`propose_loose_rois(cache, bands, *, config)`. For each band, scikit-maad
`create_mask` runs hysteresis ROI detection on the in-band PCEN
spectrogram using percentile thresholds `seed_percentile` and
`grow_percentile` *computed per recording* (so absolute value levels don't
have to be calibrated across species). `select_rois` extracts bounding
boxes; loose duration/bandwidth filters keep biological-scale events.

### 5. Shape-feature clustering

`shape_cluster_refine(loose_rois, cache, *, min_cluster_size)`. Per band,
computes 26-D Gabor shape descriptors via `maad.features.shape_features`
(requires `util.format_features` to populate pixel-coord columns) plus
intensity centroid via `maad.features.centroid_features`. The combined
feature matrix is z-standardized and clustered with
`HDBSCAN(min_cluster_size=shape_cluster_min_size, allow_single_cluster=True)`.

Each non-noise cluster becomes a candidate motif family that flows through
to EM as `motif_cluster = band_id * 100 + shape_cluster_id`. Noise ROIs
(label `-1`) are dropped. This replaces both the old `(log dur, log bw)`
KDE mode filter and the post-EM patch sub-clustering in a single step —
multi-template emission per band is driven entirely by HDBSCAN on
shape-feature space.

`tables/shape_features.csv` carries one row per loose ROI with all shape +
centroid columns and the assigned cluster id (or `-1` for noise).

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
  `bimodal_split` of the kept-score distribution (GMM(2) crossover → Otsu
  → `fallback_quantile` if unimodal), with peak distance
  `max(0.12, 0.60 * template_duration_sec)`.
- **M-step**: each kept peak is re-anchored at the *first* frame whose
  band-mean envelope exceeds `anchor_onset_fraction * max(envelope)` inside
  the support window (`_refine_anchor_via_envelope`). Re-median the patches
  to form the new template.
- Stop when `1 − cosine(new, prev) < template_change_eps` after at least one
  iteration, or after `max_em_iter`. Drop families with `< min_repetitions`
  kept peaks.

`em_trace.csv` records per-iteration `band_id, shape_cluster_id, n_kept,
threshold, method, cosine_to_prev`.

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
sharing x) per recording, matching the manual pipeline's layout.

## Configuration reference

### `BandDiscoveryConfig` (`auto_motif/band.py`)

| Knob | Default | Rationale |
|---|---|---|
| `k_band_mad` | `2.0` | Primary threshold: `med + 2*MAD` ≈ p98 of consensus |
| `k_band_mad_lower` | `1.0` | Secondary (hysteresis) threshold for grow phase |
| `min_band_width_hz` | `150.0` | ~10 FFT bins at sr=32k, nfft=2048 |
| `max_bands` | `3` | Matches observed primary / secondary / chip structure |
| `band_pad_hz` | `40.0` | One FFT bin each side |
| `fallback_min_recordings` | `2` | Need ≥2 recordings agreeing to promote a non-consensus band |
| `fallback_overlap_fraction` | `0.5` | IoU threshold for "same band" across recordings |

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
| `fallback_quantile` | `0.95` | EM bimodal-split fallback when unimodal |
| `anchor_onset_fraction` | `0.30` | First envelope frame ≥ 30% of peak; aligns peak_time with curated leading-edge |
| `ribbit_pulse_rate_hz` | `(0.3, 30.0)` | Pulse-rate window for `opensoundscape.ribbit` |
| `ribbit_clip_duration_sec` | `10.0` | RIBBIT analysis clip length per recording |
| `ribbit_score_min` | `None` | Drop bands below score (None = report only) |
| `min_activity_quantile` | `None` | Drop recordings below BI quantile (None = report only) |

## CLI

```bash
python -m data_analysis.auto_motif.mining \
  --label 22967 \
  --curated artifacts/marbled_frog/motif_boxes/curated/tables/curated_motif_boxes.csv \
  --output-dir artifacts/marbled_frog/motif_mining_auto  # optional; defaults via LABEL_TO_DIR
```

`LABEL_TO_DIR` (in `mining.py`) maps known labels to a species directory
name. Unknown labels fall back to `artifacts/<label>/motif_mining_auto/`.

## Output schema

`artifacts/<species_dir>/motif_mining_auto/`:

| Path | Schema / contents |
|---|---|
| `summary.md` | Auto-generated run summary; lists tables/plots/templates with discovered bands. |
| `tables/recording_activity.csv` | `filename, bioacoustics_index` from `maad.features.bioacoustics_index` (pre-ranking). |
| `tables/auto_bands.csv` | `band_id, fmin_hz, fmax_hz, bandwidth_hz, discovery_method, ribbit_score_max, ribbit_score_p95, ribbit_score_median, n_clips_scored`. |
| `tables/auto_band_consensus.csv` | `filename, freq_hz, z_normalized`; `filename == "__consensus__"` is the median row. |
| `tables/loose_rois.csv` | Per-band hysteresis ROIs (loose dur/bw filter). |
| `tables/shape_features.csv` | Per-ROI 26-D Gabor shape + centroid features and HDBSCAN cluster (-1 = noise). |
| `tables/em_trace.csv` | `band_id, shape_cluster_id, iter, n_kept, threshold, method, cosine_to_prev`. |
| `tables/auto_templates_metadata.csv` | One row per accepted family: band/shape-cluster IDs, threshold (EM), support, etc. |
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
| `plots/{stem}_band_envelope_vs_template_events.png` | 2-panel diagnostic per recording. |
| `plots/{stem}_full_recording_events.png` | Full focal recording with auto event boxes. |
| `templates/family_XXXX_template.npz` | Fixed-support template arrays + support dict. |

## Toggles and when to flip them

- **`shape_cluster_min_size`** controls how aggressively HDBSCAN splits ROIs
  within a band. Raise (e.g. `8–12`) when bands are noisy and you want only
  large, coherent clusters surviving. Lower (e.g. `3–4`) for species with
  small but distinct call modes.
- **`ribbit_score_min`** — set a numeric threshold (e.g. `0.05`) to drop
  bands that don't carry periodic structure. Default `None` reports scores
  without filtering — inspect `auto_bands.csv` first to pick a threshold.
- **`min_activity_quantile`** — set (e.g. `0.25`) to drop the bottom 25% of
  recordings by `bioacoustics_index` before caching their PCEN spectrograms.
  Default `None` reports scores only.
- **`anchor_onset_fraction` (default `0.30`)**: lower (e.g. `0.10–0.20`) for
  species where curated peak_time is the *very* leading edge of a call;
  higher (e.g. `0.40–0.50`) for impulsive calls where onset ≈ argmax.
- **`threshold_floor`**: drop to `0.40` if recall is the bottleneck and the
  remaining curated boxes score in the 0.55–0.65 range. Expect a precision
  hit.
- **`fallback_min_recordings`**: lower to `1` (effectively "any one
  recording") for very rare components; raise to `3+` when you have many
  recordings and want stricter agreement.
- **`band.k_band_mad_lower`**: tune the secondary hysteresis threshold. The
  default `1.0` extends bands by ~1 FFT bin until consensus falls below
  `med + 1*MAD`. Lower it (e.g. `0.5`) to extend further; raise (e.g. `1.5`)
  to keep bands tight.
- **`band.max_bands`**: raise above `3` for species with more than three
  distinct call components, but pay attention to within-band consolidation
  artifacts when bands overlap.

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

If a second species needs these, lift them into `common/motif_template.py`
(currently they live under `marbled_frog/` for historical reasons but the
implementations have no Marbled Frog-specific logic).

## Marbled Frog results (sanity check)

After the 2026-05-13 library-tool migration (PCEN + shape_features +
RIBBIT + bioacoustics_index): **28/33 (84.8%) consolidated curated
recovery** vs. the prior 25/33 (75.8%) baseline. Per-component:

| Component | Curated | Recovered | Rate | Median best score |
|---|---|---|---|---|
| chip_1k_1p8k | 5 | 5 | 1.00 | 0.80 |
| primary | 4 | 4 | 1.00 | 0.73 |
| primary_independent | 10 | 10 | 1.00 | 0.64 |
| secondary | 4 | 0 | 0.00 | 0.73 |
| secondary_independent | 10 | 9 | 0.90 | 0.73 |

The remaining miss is the `secondary` component: detections exist with
score ≥ 0.73 but get cross-family-consolidated into the temporally
adjacent primary-band event. This is a consolidation policy issue, not a
detection failure. RIBBIT scores were uniformly low (~0.08–0.14) for
Marbled Frog — it doesn't have a regular pulse-rate signature — so the
score is reported but should not be used as a hard filter here.

## Limitations and follow-ups

- **Frequency-IoU strictness in recovery eval**. Bands that fire on a
  call's harmonic content but don't overlap the curated fundamental are
  scored as misses. Consider a harmonic-aware matcher in
  `common/curated_eval.py`.
- **Secondary-component consolidation**. Cross-family consolidation
  collapses temporally adjacent secondaries into primary events. Worth
  revisiting `consolidate_peak_distance_sec` or adding a band-keep
  override.
- **No per-recording band addition**. Phase 4's fallback adds a band only
  if a *majority of recordings* support it. Species with one outlier
  recording carrying a rare component aren't yet rescued.
- **Cache memory**. `load_focal_z` holds wideband spectrograms for every
  recording in RAM. For species with many long recordings, lazy iteration
  + on-disk caches may be needed.
