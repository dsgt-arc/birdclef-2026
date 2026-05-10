# Notebook: Marbled White-lipped Frog Motif Mining

## Header

| Field | Value |
|---|---|
| Date | 2026-05-10 |
| Status | Design doc — current pipeline + planned redesign |
| BirdCLEF label | `22967` |
| Species | Marbled White-lipped Frog / *Leptodactylus elenae* |
| Code | `src/data_analysis/marbled_frog/motif_mining.py` |
| Artifacts | `artifacts/marbled_frog/motif_mining/` |

## Goal

Discover the actual repertoire of recurring motifs in `22967` train audio without pre-committing to known frequency bands. Each motif must come from a *property* (a time-frequency event with bounded duration, bandwidth, salience), not from a hand-picked band that re-encodes what we already inspected. Clusters of similar motifs are the unit of human review, so we label families instead of individual boxes.

## Current pipeline (as of today)

```text
audio
  → STFT power → dB
  → per-frequency robust z-score (median/MAD with floor)
  → clip to [0, 8] → Gaussian smooth (σ_freq=1.5, σ_time=1.0)
  → 8-connected components above z ≥ 2.5
  → property gates (duration, bandwidth, peak_z, area)
  → bbox = component extent
  → normalized 64×48 crop per box
  → StandardScaler → PCA → KMeans(k=8)
  → cluster grids + per-cluster summary
```

Property gates currently in use:

```text
z_threshold        = 2.5
min/max duration   = 0.03 / 1.5 s
min/max bandwidth  = 80 / 2200 Hz
min peak z         = 3.5
min area (pixels)  = 8
```

Current run: **280 components across 8 recordings** (range 2 to 98 per recording), clustered into 8 KMeans groups. Per-recording counts:

```text
iNat1448422: 18    iNat1697350: 17   iNat1698104:  6
iNat1698105: 29    iNat1698107: 98   iNat321313: 19
iNat56387:    2    iNat911059:  91
```

## Review findings

Human review of `cluster_XX_grid.png` flagged two systematic failures:

1. **Boxes too small / subset of the real component.** A motif's tail dips below `z = 2.5` and gets clipped; adjacent bright spots inside the same biological event split into two components. Showed up most in clusters 02 (bottom rows), 04, and several "bad" clusters.

2. **Mono-recording clusters.** Cluster 03 came entirely from `iNat911059` (which holds 91/280 candidates). PCA over 64×48 normalized pixel crops is partly clustering on recording-level texture (background spectrum, dynamic range) rather than motif shape.

Both are upstream of clustering. Reviewing labels on clipped extents would be wasted: the boxes would change after the segmentation fix and the cluster centroids would shift.

## Redesign: lean on scikit-maad

[`scikit-maad`](https://scikit-maad.github.io/) is the canonical Python library for bioacoustic ROI detection (Ulloa et al. 2021, MEE; pure-Python, only depends on numpy/scipy/pandas/matplotlib/scikit-image; v1.5.2 released Feb 2026, actively maintained). It already implements the exact primitives we'd otherwise hand-roll.

### Segmentation (replaces our `detect_components`)

```python
from maad import sound, rois, util

# 1. dB spectrogram, lifted to non-negative, then smoothed.
Sxx_dB = util.power2dB(Sxx) + 96
Sxx_dB_smooth = sound.smooth(Sxx_dB)

# 2. Double-threshold + hysteresis propagation.
#    Two modes:
#      'relative'  → bin_std (high) + bin_per (low fraction of high)
#      'absolute'  → bin_h, bin_l in dB
im_bin = rois.create_mask(
    Sxx_dB_smooth,
    mode_bin='absolute',
    bin_h=6.0,   # high seed threshold (≈ 4 sigma above noise)
    bin_l=3.0,   # low growth threshold
)

# 3. Connected components + area filter; rois is a pandas DataFrame.
im_rois, rois_df = rois.select_rois(im_bin, min_roi=8, max_roi=None)

# 4. Convert pixel coords → (t, f) seconds/Hz, in place.
rois_df = util.format_features(rois_df, tn, fn)
# Columns now include: labelID, min_t, max_t, min_f, max_f, plus pixel coords.
```

`create_mask` is exactly the hysteresis primitive `pi` recommended; using it removes our hand-rolled `binary_propagation` + `label` calls. The `relative` mode internally does `median + bin_std * MAD`, which is what our `robust_excess_spectrogram` already produces, so we have a choice between letting maad recompute robust statistics on the dB image or feeding it our pre-z-scored map and using `absolute` mode. Either works; `absolute` on our z-map is more transparent because thresholds are in z units.

**Pre-step we still need to add: morphological closing.** Hysteresis alone won't merge two bright spots separated by a valley below the low threshold. Add `scipy.ndimage.binary_closing(low_mask, structure=ones((3,1)))` (3 time bins × 1 freq bin) before propagation. SE size will be tuned visually.

### Features (replaces our hand-rolled `box_features`)

```python
from maad import features

shape_df, params = features.shape_features(
    Sxx, resolution='med', rois=rois_df,
)
centroid_df = features.centroid_features(Sxx, rois=rois_df)
```

`shape_features` runs a 2D Gabor filter bank (θ × frequency × scale) and returns ~52 standardized per-ROI features under the `shp_*` prefix. Together with `centroid_features` (intensity centroid), this gives us a much richer, standardized morphological descriptor than the four numbers we compute today.

### What we keep from our current code

- `robust_excess_spectrogram` (we may still want our own z-map for visualization).
- `crop_patch` — but we'll add **per-row spectral normalization** (`pi`'s suggestion: divide each crop by its row-wise median) to strip recording-level texture before PCA.
- `cluster_candidates`, `plot_candidate_embedding`, `plot_cluster_grid` — restructure to consume the new ROI table.

### What we drop

- `ComponentGate` dataclass + the post-hoc duration/bandwidth/peak/area gate loop → `min_roi`/`max_roi` in `select_rois` plus optional column filters on the returned DataFrame.
- `box_features` (replaced by maad shape + centroid features).
- The `ComponentBox` dataclass; the maad `rois_df` is already the right schema.

## Clustering plan

Today we do `StandardScaler → PCA → KMeans` on flattened normalized pixel crops. The field-standard for unsupervised animal-vocalization repertoire clustering (Sainburg et al. 2020, *PLOS Comp Bio*; AVGN; VocalPy) is **UMAP + HDBSCAN** on aligned syllable spectrograms or learned embeddings. KMeans forces equal-size clusters and breaks badly when one recording dominates; HDBSCAN does density-aware clustering with a `noise` label, which is what we want when most candidates are real-but-similar and a few are outliers.

Three feature-representation options, in order of escalation:

1. **Spectrally-normalized pixel crops.** Cheapest fix: row-divide each crop by its median spectrum, then re-run pixel-PCA + (KMeans → HDBSCAN). Tests whether texture domination was the only issue.
2. **scikit-maad shape + centroid features** (~50-d, standardized, interpretable). UMAP(5) → HDBSCAN(min_cluster_size=5). Loses crop morphology but gains interpretability and recording-invariance.
3. **Perch v2 embeddings per ROI.** Per BirdCLEF / bioacoustic-ML literature (Ghani et al. 2023; Frontiers Ecol Evol 2024), Perch ≫ pixel features for animal-call clustering. We already have these precomputed for the focal/train-audio set:

   ```text
   /mnt/data/scratch/birdclef-2026/inference/perch-v2-tf/train_audio/embeddings.parquet
   ```

   These are 5-second windowed embeddings, so each ROI joins to the embedding window containing its `peak_time_sec`. Coarser than per-frame mean-pooling but free. UMAP(5) → HDBSCAN.

Decide between (1), (2), (3) after re-running with fixed segmentation. (1) is the prediction made by `pi` (that recording bias is downstream of clipped boxes). (3) is the field-standard upgrade.

## Proposed sequence

1. Add `scikit-maad` to `pyproject.toml` (`uv pip install scikit-maad` — pure-Python, ~scikit-image-tier deps).
2. Rewrite `detect_components` to use `sound.smooth → ndimage.binary_closing → rois.create_mask → rois.select_rois → util.format_features → features.shape_features + features.centroid_features`. Keep the output schema compatible with the current `candidates.csv`.
3. Re-run, re-cluster with current pixel-PCA + KMeans. Compare grids: are the "too small" boxes resolved? Are clusters less mono-recording?
4. If recording bias persists, apply per-row spectral normalization to crops. Re-cluster.
5. If still bad, switch features → maad shape/centroid OR Perch-window embeddings; clustering → UMAP + HDBSCAN.
6. Once clustering is stable, add `tables/candidate_review.csv` (one row per candidate, keyed by `(filename, peak_time_sec, peak_freq_hz, area_pixels)`) with `verdict`/`notes` columns the user fills in by hand. Reruns merge on the stable key.

## Open questions

- **Closing SE size.** `pi` suggested disk radius 2-3 time × 1-2 freq; subagent's MAAD-style recipe used 3 time × 1 freq. Tune visually after first run.
- **Hysteresis thresholds.** `bin_h ≈ 6 dB`, `bin_l ≈ 3 dB` are MAAD defaults in dB on the maad-smoothed dB image. We need to translate to our robust-z map (currently `seed ≈ z=3.5`, `mask ≈ z=1.5`). Sanity-check that the new thresholds reproduce known curated counts (iNat1448422: 4 low + 4 secondary; iNat1697350: ~10+10; iNat1698104: 5 chips).
- **Whether to keep our own `robust_excess_spectrogram`.** maad has its own `remove_background_along_axis` + `median_equalizer` which is roughly equivalent. We probably want both: our z-map for visualization (consistent with curated/review plots), maad's variant if it segments better.
- **Whether to add wavelet ROI detection** (`rois.find_rois_cwt`) for chips with known duration/bandwidth priors — strictly optional; might surface chip-family candidates the connected-component path misses.

## What this won't fix

- Recordings with no real motifs (iNat56387 — quietest recording, max z ≈ 3.7) will still produce 0-2 candidates. Acceptable.
- Soundscape / overlap: this pipeline assumes events are spatially separated in time-frequency. Overlapping calls will be merged into one component, which is fine for motif discovery but bad if we ever want per-call labels. Out of scope here.
- We are not training a learned detector. DCASE Task 5 few-shot prototypical networks would be the next-level upgrade if we ever want one; not needed at 280 events.
