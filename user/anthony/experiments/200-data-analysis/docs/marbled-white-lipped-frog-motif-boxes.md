# Notebook: Marbled White-lipped Frog Motif Boxes

## Header

| Field | Value |
|---|---|
| Date | 2026-05-10 |
| BirdCLEF label | `22967` |
| Species | Marbled White-lipped Frog / *Leptodactylus elenae* |
| Main code | `src/data_analysis/marbled_frog/motif_boxes.py` |
| Review code | `src/data_analysis/marbled_frog/motif_review.py` |
| Main artifacts | `artifacts/marbled_frog/motif_boxes/` (with `curated/` subset), `artifacts/marbled_frog/motif_review/` |

## Purpose

This thread explored how to extract recurring Marbled White-lipped Frog motifs from train-audio spectrograms. The aim is not yet final ground-truth annotation. The aim is to create useful motif evidence/crop hints that can support manual review, pseudo-labeling, or a future learned detector.

## Starting point

We began with visual inspection of several train recordings and noticed repeated motif families:

- `iNat1448422`: about 4 short low clicks in the 0-5s window, with a lighter secondary/harmonic-like component.
- `iNat1697350`: about 10 repeated paired-looking events, with lower primary/tail structure and a broad upper sweep.
- `iNat1698104`: about 5 rising chips around 1-1.8 kHz.

The early boxes were generated from simple band-envelope peak detection:

```text
STFT/dB spectrogram
→ robust per-frequency normalization
→ band envelope
→ find peaks
→ envelope-derived time bounds
→ frequency-profile-derived fmin/fmax
```

This was useful, but boxes were often too rectangular, too band-dependent, or needed hand tuning.

## What we tried

### 1. Manual motif-box workflow

Code:

```text
src/data_analysis/marbled_frog/motif_boxes.py
```

Outputs land under `artifacts/marbled_frog/motif_boxes/{template_matches,curated}/`. The module produces prototype primary/secondary boxes per recording. We iteratively adjusted:

- primary band from roughly `900-1300 Hz` down to `600-1300 Hz`,
- secondary band for `1448422` to `1500-2300 Hz`,
- broader `1697350` secondary validation band to `1000-3000 Hz`,
- `1698104` custom chip band to `1000-1800 Hz`.

### 2. Template matching experiment

We briefly created a manual template-matching workflow. It treated a few hand-picked spectrogram crops as templates and matched them across all recordings.

Result: deleted/abandoned.

Why:

- It drifted away from the more promising motif-box workflow.
- Template matches were useful conceptually, but the user preferred returning to explicit motif boxes.
- It also risked over-anchoring on a few source crops.

### 3. Motif mining via connected components

Code:

```text
src/data_analysis/marbled_frog/motif_mining.py
```

Artifacts:

```text
artifacts/marbled_frog/motif_mining/
```

Each motif candidate is exactly one 8-connected component in the robust-excess-z spectrogram. Components are gated on physical event properties rather than hand-picked frequency bands:

```text
z_threshold        binarization cutoff over robust-excess-z
min/max duration   bbox time-extent must fall in this range
min/max bandwidth  bbox frequency-extent range
min peak z         component must contain at least one pixel this bright
min area (pixels)  minimum component size
```

This replaced an earlier three-band detector bank (`low_primary_600_1300`, `rising_chip_1000_1800`, `broad_upper_1000_3000`) that re-encoded the curated subset's hand inspection. The component-based detector treats *all* recordings the same way, so any band structure that appears in clusters comes from the data, not from the detector.

Each component yields one bounding box (its time-frequency extent), plus features `peak_z`, `mean_z`, `area_pixels`, `fill_ratio`, `freq_centroid_hz`, `freq_slope_hz_per_s`. Crops are then PCA-embedded and KMeans-clustered as before.

### 4. Claude Opus review

We asked Claude Opus for critique. Main points:

- Good: tiered framing, genus-level Perch signal, multi-component call model, cluster-level review.
- Fragile: MAD normalization could explode when per-frequency MAD is near zero.
- Fragile: too much seed-file anchoring.
- Fragile: detector bands can determine frequency bounds and clusters.
- Recommendation: use robust z-score floor/clipping, 2D smoothing, known-count peak selection, connected components, and tiered pseudo-labeling.

### 5. Signal-derived box refinement

Implemented in:

```text
src/data_analysis/marbled_frog/motif_boxes.py
```

Key upgrades:

```text
MAD floor + z-score clipping
2D Gaussian smoothing
band envelope + tonality score
expected-count peak selection
connected-component-derived box extents
```

This made the boxes more signal-derived and less purely based on hand-padded frequency bands.

Known count priors currently used:

```text
iNat1448422: low primary 4, secondary 4
iNat1697350: low primary 10, broad upper 10
iNat1698104: mid/rising chip 5
```

### 6. Curated current-best subset

Code: same module — see `CURATED_ANNOTATIONS` and `emit_curated()` in `src/data_analysis/marbled_frog/motif_boxes.py`. Emitted automatically at the end of `run()`. Each curated entry has its own detector (seed-window / independent bands / single-component chip) defined inline in `emit_curated`.

Artifacts:

```text
artifacts/marbled_frog/motif_boxes/curated/
├── manifest.csv
├── plots/
│   ├── iNat1448422_0-5s_curated.png
│   ├── iNat1697350_curated.png
│   └── iNat1698104_curated.png
└── tables/
    ├── iNat1448422_0-5s_motif_boxes.csv
    ├── iNat1697350_independent_band_motif_boxes.csv
    ├── iNat1698104_1k_1p8k_chip_boxes.csv
    └── curated_motif_boxes.csv
```

Per-recording tables hold the raw detector output for that curated entry. `curated_motif_boxes.csv` is the union, tagged with `annotation_set`, `motif_family`, `annotation_status`, `annotation_notes`. Plots re-render the spectrograms with standardized primary/secondary validation bands.

Current curated box counts:

```text
iNat1448422:
  primary:   4
  secondary: 4

iNat1697350:
  primary_independent:   10
  secondary_independent: 10

iNat1698104:
  chip_1k_1p8k: 5
```

### 7. Full recording motif review

Code:

```text
src/data_analysis/marbled_frog/motif_review.py
```

Artifacts:

```text
artifacts/marbled_frog/motif_review/
```

This renders one review plot per `22967` train recording using a fixed detector bank:

```text
green  = low_primary_600_1300
yellow = mid_chip_1000_1800
cyan   = broad_upper_1000_3000
```

The key idea is that these are **proposal channels**, not biological classes.

For `iNat1697350`, visual review showed:

```text
green = good
cyan = fine
yellow = not a real motif there
```

So `iNat1697350` now suppresses yellow/mid-chip proposals in the review output.

## Current interpretation

We do **not** know there are exactly 3 motif classes. The three detector colors are just useful proposal channels:

- low primary / lower tail,
- mid rising chip,
- broad upper / secondary.

The curated recordings support at least these motif families:

```text
low_click_with_secondary
broad_upper_sweep_with_low_tail
rising_chip_1k_1p8k
```

But these should remain provisional until we validate against hard negatives and more recordings.

## What not to overclaim

These boxes are not final biological ground-truth annotations. They are:

```text
motif evidence
crop hints
proposal boxes
review artifacts
```

They should not yet be used as hard labels for species detection without further validation.

## Possible next directions

### Short-term

- Inspect the full recording review plots.
- Mark which detector channels are useful for each recording.
- Add a small anchor table of accepted/rejected boxes.
- Pair primary and secondary events in `1697350` by time proximity.

### Medium-term

- Add Perch genus-logit validation at event/window level.
- Compare candidate boxes against hard-negative `Leptodactylus` recordings.
- Convert accepted boxes into crop-level training examples.

### Longer-term

- Try YOLO as a motif proposal detector, not as a final species detector.
- Use current curated boxes as weak starting labels.
- Validate YOLO detections visually and with Perch/template/context signals.

## Current recommendation

Treat the current motif boxes as good enough for review and crop proposals. Before training anything serious, create a small manually accepted/rejected anchor set and use that to calibrate thresholds or a learned detector.
