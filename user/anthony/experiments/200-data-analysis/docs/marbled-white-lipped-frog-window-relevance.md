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

A window is `likely_relevant` if:

```text
max Leptodactylus Perch logit >= 8.5
AND
1-3 kHz mel energy fraction >= 0.5
```

Otherwise it is `likely_not_relevant`.

This is **not ground truth**. It is a pseudo-label for inspection and ranking.

## Counts

```text
relevance_label|n_windows|mean_max_genus_logit|mean_mel_1_3k_energy_frac
likely_not_relevant|17|6.98|0.51
likely_relevant|16|9.12|0.87
```

## Generated plots

| Plot | Description |
|---|---|
| `window_relevance_pacmap.png` | Perch embedding PaCMAP of all 33 windows, colored by weak relevance label |
| `spectrogram_samples_relevant_vs_not.png` | sample mel spectrograms: likely relevant vs likely not relevant |

## Main visual takeaway

The relevant spectrogram samples show repeated rising frog-like calls concentrated inside the **1-3 kHz** prior band, often with strongest visible energy around **1-2 kHz** plus harmonics/upper structure.

The not-relevant samples include windows with weaker or more contaminated structure: more broadband/horizontal background energy, partial windows, lower max-genus support, or lower 1-3 kHz concentration.

The PaCMAP plot shows that the weak labels are not perfectly separable, but the high-confidence relevant windows form recognizable subgroups, including a distinct `iNat1698107` cluster/outlier with very strong repeated call structure.

## Artifacts

```text
window_relevance_labels.csv
window_relevance_pacmap.csv
window_relevance_pacmap.png
spectrogram_samples_relevant_vs_not.png
summary.md
```

## Caveat

These labels should be treated as a triage layer. Before training with them as hard targets, manually verify a small sample of windows and possibly introduce an `uncertain` label for borderline cases.
