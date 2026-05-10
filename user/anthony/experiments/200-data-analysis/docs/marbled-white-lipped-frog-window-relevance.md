# Notebook: Marbled White-lipped Frog Window Relevance Labels

## Header

| Field | Value |
|---|---|
| Date | 2026-05-10 |
| BirdCLEF label | `22967` |
| Species | Marbled White-lipped Frog / *Leptodactylus elenae* |
| Analysis code | `src/data_analysis/marbled_frog_window_relevance.py` |
| Input analysis | `artifacts/species_frequency_perch_22967/window_scores_and_spectral_features.csv` |
| Output directory | `artifacts/marbled_frog_window_relevance/` |

## Heuristic label rule

Each 5-second train-audio window is assigned a weak relevance label using the prior from the mel/logit analysis:

```text
likely_relevant =
    max Leptodactylus Perch logit >= 8.5
    AND
    1-3 kHz mel energy fraction >= 0.5
```

Otherwise it is `likely_not_relevant`.

This is **not ground truth**. It is a pseudo-label for inspection and ranking.

Color convention:

```text
red  = scalar likely_relevant
blue = scalar likely_not_relevant
```

## Counts

```text
relevance_label|n_windows|mean_max_genus_logit|mean_mel_1_3k_energy_frac
likely_not_relevant|17|6.98|0.51
likely_relevant|16|9.12|0.87
```

## Cluster inspection

The manually selected PaCMAP region around roughly `(-4, 0)` is used as an inspection cluster, not as final truth. A robust PaCMAP/MAD boundary around that region contains 21 windows:

```text
inside selected cluster:
  scalar likely_not_relevant: 15
  scalar likely_relevant:      6
```

The 12 outside-cluster windows split cleanly into two PaCMAP-space k-means groups:

```text
outside k-means cluster 0: 7 windows, all iNat1698107
outside k-means cluster 1: 5 windows, all iNat1698105
```

## Organized artifacts

Artifacts are now grouped to avoid top-level sprawl:

```text
artifacts/marbled_frog_window_relevance/
├── summary.md
├── plots/
│   ├── window_relevance_pacmap.png
│   ├── inside_cluster_spectrogram_grid.png
│   ├── outside_cluster_kmeans_pacmap.png
│   ├── outside_cluster_kmeans_spectrogram_grid.png
│   ├── best_window_per_recording_spectrogram_grid.png
│   ├── peak_frequency_by_recording.png
│   └── spectrogram_samples_relevant_vs_not.png
├── spectra/
│   ├── iNat1698107_frequency_spectra.png
│   ├── iNat911059_frequency_spectra.png
│   └── ... one per train recording
└── tables/
    ├── window_relevance_labels.csv
    ├── window_relevance_pacmap.csv
    ├── inside_cluster_windows.csv
    ├── outside_cluster_kmeans_windows.csv
    ├── all_recording_peak_summary.csv
    ├── best_window_per_recording.csv
    └── ... boundary/member tables
```

## Best window per recording

The script also creates a `relevant_vs_not`-style grid with the highest relevance-score window from each of the 8 train recordings:

```text
artifacts/marbled_frog_window_relevance/plots/best_window_per_recording_spectrogram_grid.png
artifacts/marbled_frog_window_relevance/tables/best_window_per_recording.csv
```

Selected windows:

```text
filename|window|label|cluster
22967/iNat1448422.ogg|0-5s|likely_relevant|inside selected cluster
22967/iNat1697350.ogg|0-5s|likely_relevant|inside selected cluster
22967/iNat1698104.ogg|0-5s|likely_not_relevant|inside selected cluster
22967/iNat1698105.ogg|15-20s|likely_relevant|outside k=1
22967/iNat1698107.ogg|30-35s|likely_relevant|outside k=0
22967/iNat321313.ogg|0-5s|likely_not_relevant|inside selected cluster
22967/iNat56387.ogg|5-10s|likely_relevant|inside selected cluster
22967/iNat911059.ogg|15-20s|likely_relevant|inside selected cluster
```

## Spectral peak analysis

The script now generates spectroscopy-style line plots for all 8 train recordings. For each 5s window it computes mean log-power from 500-3500 Hz and detects the top spectral peaks.

Median strongest peak by recording:

```text
filename|median_peak1_hz|interpretation
22967/iNat321313.ogg|1000.0|low-frequency/low-support regime
22967/iNat1448422.ogg|1179.7|mixed lower regime
22967/iNat56387.ogg|1421.9|central motif, high repetition/cadence variation
22967/iNat1698104.ogg|1468.8|central motif, short file
22967/iNat911059.ogg|1484.4|central motif plus secondary/harmonic-like component
22967/iNat1698107.ogg|1515.6|clean prototype group
22967/iNat1697350.ogg|1992.2|higher-frequency candidate regime
22967/iNat1698105.ogg|2796.9|distinct high-frequency outside-cluster regime
```

## Main visual takeaway

The scalar relevance rule captures many high-logit/high-band-energy windows, but the spectrograms and peak plots show multiple acoustic regimes. The cleanest prototype is `iNat1698107`; `iNat56387` and `iNat911059` look plausible but more complex, with cadence or multi-component structure.

These labels should remain a triage layer. Before training with them as hard targets, manually verify representative windows from each discovered regime and possibly introduce tier labels such as `high_confidence_relevant`, `candidate_relevant`, and `needs_review`.
