# Case Study: Marbled White-lipped Frog

## Header

| Field | Value |
|---|---|
| Date | 2026-05-10 |
| BirdCLEF label | `22967` |
| Species | Marbled White-lipped Frog / *Leptodactylus elenae* |
| Class | Amphibia |
| Train-audio files | 8 |
| Train-audio 5s windows | 33 |
| Main artifacts | `artifacts/marbled_frog/frequency_perch/`, `artifacts/marbled_frog/window_relevance/` |
| Main code | `src/data_analysis/marbled_frog/frequency_perch.py`, `src/data_analysis/marbled_frog/window_relevance.py` |

## One-line takeaway

For Marbled White-lipped Frog, Perch is useful mostly as a **genus-level detector**, while the best current interpretation comes from combining **max *Leptodactylus* logit**, **1-3 kHz energy**, **PaCMAP/embedding structure**, and now **spectral peak patterns**.

## Why this species

Marbled White-lipped Frog is useful as a small case study because it has:

- enough labeled soundscape support to matter: 310 labeled soundscape windows across 14 soundscape files,
- limited but nonzero train-audio support: 8 files / 33 Perch windows,
- multi-site labeled-soundscape presence,
- obvious examples where weak file-level train labels are not enough to know which 5s windows are actually relevant.

## Step 1: Start from train-audio windows

The regular training metadata says these 8 files have primary label `22967`. Perch embeddings/logits split them into 33 five-second windows.

But `train.csv` only gives file-level labels. It does not say which 5s windows contain the frog call. So every per-window relevance label here is a weak/pseudo label unless manually verified.

The metadata has a `rating` column, but all 8 recordings for this species are iNaturalist examples with `rating=0.0`, which appears to mean **unrated/no quality metadata**, not necessarily bad quality.

## Step 2: Check Perch logits

The direct Perch species logit for *Leptodactylus elenae* is weak. Across the 33 windows, the top mean Perch taxa are nearby congeners:

```text
rank|perch_label|mean_logit
1|Leptodactylus fragilis|7.19
2|Leptodactylus fuscus|6.78
3|Donacobius atricapilla|6.03
```

Interpretation:

- Do **not** use the direct *L. elenae* Perch logit as the main relevance signal.
- Use the maximum logit over Perch labels beginning with `Leptodactylus` as a genus-level frog evidence score.

This makes biological/modeling sense: Perch hears something strongly *Leptodactylus*-like, but does not resolve the exact BirdCLEF species reliably.

## Step 3: Estimate a useful frequency band

Using librosa mel spectra, I compared windows with high vs low max-*Leptodactylus* logits.

The enriched bands in high-genus-logit windows are:

```text
band_hz|high_minus_low|high_div_low
2000-3000|+0.204|3.69x
1500-2000|+0.147|3.19x
1000-1500|+0.090|1.38x
```

Working frequency prior:

```text
broad:   1-3 kHz
strong:  1.5-3 kHz
```

This is not a time-frequency bounding box. It is a prior for inspection/filtering.

## Step 4: Initial weak relevance rule

The first window-level pseudo-label rule was:

```text
likely_relevant =
    max Leptodactylus Perch logit >= 8.5
    AND
    1-3 kHz mel energy fraction >= 0.5
```

This produced:

```text
likely_relevant:     16 windows
likely_not_relevant: 17 windows
```

Color convention in plots:

```text
red  = scalar likely_relevant
blue = scalar likely_not_relevant
```

These scalar labels are only triage labels; they are not final truth.

## Step 5: PaCMAP shows several acoustic regimes

The PaCMAP view shows one manually selected cluster around approximately:

```text
PaCMAP seed center ≈ (-4, 0)
manual seed radius = 3
```

Using the seed points, I computed a robust 2D PaCMAP/MAD boundary:

```text
centroid ≈ (-3.83, -0.01)
radius   ≈ 3.38  using median distance + 2 * 1.4826 * MAD
members  = 21 windows
```

This selected cluster is **not a ground-truth positive or negative class**. It is a visually coherent acoustic regime. It contains both scalar-red and scalar-blue windows:

```text
inside selected cluster:
  scalar likely_not_relevant: 15
  scalar likely_relevant:      6
```

The windows outside this selected cluster split cleanly into two k-means groups:

```text
outside k-means cluster 0: 7 windows, all 22967/iNat1698107.ogg
outside k-means cluster 1: 5 windows, all 22967/iNat1698105.ogg
```

The `iNat1698107` group is the cleanest high-confidence repeated-call prototype seen so far. The `iNat1698105` group is also strongly *Leptodactylus*-like but has a different spectral emphasis.

## Step 6: Inspect inside and outside clusters by spectrogram

Generated inspection grids:

```text
artifacts/marbled_frog/window_relevance/plots/inside_cluster_spectrogram_grid.png
artifacts/marbled_frog/window_relevance/plots/outside_cluster_kmeans_spectrogram_grid.png
artifacts/marbled_frog/window_relevance/plots/best_window_per_recording_spectrogram_grid.png
```

The best-window-per-recording grid selects the highest relevance-score window from each of the 8 train recordings, making it easier to compare the most target-like motif per source file.

Inside the selected cluster, several recordings show plausible frog-like motifs but with different texture:

- `iNat56387`: shape looks plausible, but the motif happens much more frequently; possible multi-individual/chorus or cadence mismatch.
- `iNat911059`: one main motif plus a secondary motif; likely multi-component or partly harmonic structure.
- `iNat321313`: lower-support windows concentrated near ~1 kHz.

So the inside cluster is not simply false positives. It is better treated as a **mixed/candidate regime** requiring manual review.

## Step 7: Spectroscopy-style peak detection across all recordings

I added line-plot spectra for all 8 train recordings:

```text
artifacts/marbled_frog/window_relevance/spectra/*_frequency_spectra.png
artifacts/marbled_frog/window_relevance/tables/all_recording_peak_summary.csv
artifacts/marbled_frog/window_relevance/plots/peak_frequency_by_recording.png
```

For each 5s window, the code computes mean log-power from 500-3500 Hz and detects the top spectral peaks.

Median strongest-peak frequency by recording:

```text
filename|n_windows|median_peak1_hz|min_peak1_hz|max_peak1_hz|mean_genus|mean_band
22967/iNat321313.ogg|4|1000.0|937.5|1109.4|7.83|0.16
22967/iNat1448422.ogg|4|1179.7|1109.4|2828.1|7.17|0.36
22967/iNat56387.ogg|3|1421.9|1421.9|1421.9|8.99|0.74
22967/iNat1698104.ogg|1|1468.8|1468.8|1468.8|7.82|0.86
22967/iNat911059.ogg|5|1484.4|1437.5|1562.5|8.19|0.99
22967/iNat1698107.ogg|7|1515.6|1500.0|1531.2|9.16|0.95
22967/iNat1697350.ogg|2|1992.2|1984.4|2000.0|9.07|0.86
22967/iNat1698105.ogg|5|2796.9|2765.6|2812.5|9.42|0.71
```

This suggests several consistent but distinct peak families:

- ~1.0 kHz: low-frequency candidate/possible different acoustic regime,
- ~1.4-1.5 kHz: central motif family, including `iNat56387`, `iNat911059`, and `iNat1698107`,
- ~2.0 kHz: `iNat1697350`,
- ~2.8 kHz: `iNat1698105`, a distinct high-frequency group.

For `iNat911059`, the spectrum shows a dominant ~1.45-1.48 kHz component plus a weaker ~1.06-1.08 kHz component and sometimes a faint ~2.1 kHz component. The ~1.06/~2.14 pair is harmonic-like, but the dominant ridge is closer to ~1.5 kHz.

## Current best interpretation

There are multiple plausible tiers/regimes rather than one binary label:

### High-confidence relevant prototype

`iNat1698107`: tight outside-cluster group with seven consecutive red windows, high max-*Leptodactylus* logits, strong 1-3 kHz energy, and stable strongest peaks around ~1.50-1.53 kHz.

### Candidate relevant, central motif / cadence variation

`iNat56387`, `iNat911059`, and related windows: plausible frog-like motifs around ~1.4-1.5 kHz but with cadence/multi-component differences.

### Candidate relevant, different spectral regime

`iNat1697350` around ~2.0 kHz and `iNat1698105` around ~2.8 kHz. These have strong genus logits but look spectrally distinct from the clean `iNat1698107` prototype.

### Lower-confidence / needs review

Windows with low genus logit, weak 1-3 kHz concentration, or peaks near ~1.0 kHz only.

## Practical next rule

A better pseudo-labeling approach should be tiered:

```text
high_confidence_relevant =
    close to clean prototype group, especially iNat1698107-like windows
    AND max Leptodactylus logit high
    AND stable call-like spectral peak pattern

candidate_relevant =
    high genus logit or good 1-3 kHz call-like structure
    BUT cadence, motif, or peak family differs from the clean prototype

needs_review_or_likely_not_relevant =
    low genus support, weak band concentration, or unclear spectrogram structure
```

For training, I would not use the initial binary labels directly. I would manually review a small set from each discovered regime and then promote confirmed regimes into separate pseudo-label tiers.
