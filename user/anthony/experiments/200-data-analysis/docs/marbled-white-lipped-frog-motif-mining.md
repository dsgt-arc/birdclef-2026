# Notebook: Marbled White-lipped Frog Motif Mining

## Header

| Field | Value |
|---|---|
| Date | 2026-05-10 |
| BirdCLEF label | `22967` |
| Species | Marbled White-lipped Frog / *Leptodactylus elenae* |
| Analysis code | `src/data_analysis/marbled_frog/motif_mining.py` |
| Output directory | `artifacts/marbled_frog/motif_mining/` |

## Purpose

Manual per-recording box tuning was useful for discovering motifs, but it does not scale. This script switches to a motif-mining workflow:

```text
over-detect candidate boxes → crop normalized spectrogram patches → cluster patches → review cluster grids
```

The goal is to label **motif clusters**, not individual boxes one at a time.

## Candidate detectors

The first version uses three broad envelope detectors:

```text
low_primary_600_1300    600-1300 Hz
rising_chip_1000_1800   1000-1800 Hz
broad_upper_1000_3000   1000-3000 Hz
```

Each detector produces candidate time-frequency boxes from robust-excess spectrogram envelopes. For each candidate, the code extracts a normalized spectrogram crop and computes interpretable features such as duration, frequency bounds, centroid, slope, and peak score.

## Artifacts

```text
artifacts/marbled_frog/motif_mining/
├── summary.md
├── plots/
│   └── candidate_motif_pca.png
├── clusters/
│   ├── cluster_00_grid.png
│   ├── cluster_01_grid.png
│   └── ...
└── tables/
    ├── candidates.csv
    ├── boxes_by_cluster.csv
    └── cluster_summary.csv
```

## Current run summary

The current run generated 317 candidate motif boxes across all 8 train recordings and clustered them into 8 motif groups.

```text
cluster|n|recordings|dominant_detector|median_duration|median_fmin|median_fmax|median_centroid|median_slope
0|45|6|rising_chip_1000_1800|0.332|1000|1800|1375|73
1|84|6|low_primary_600_1300|0.268|802|1297|1084|1107
2|33|3|broad_upper_1000_3000|0.364|1000|3000|1803|-1095
3|19|1|rising_chip_1000_1800|0.380|1000|1799|1345|778
4|56|5|rising_chip_1000_1800|0.268|1000|1713|1314|2185
5|27|1|low_primary_600_1300|0.460|661|1297|1053|538
6|41|6|broad_upper_1000_3000|0.300|1000|1800|1456|1654
7|12|4|rising_chip_1000_1800|0.732|1000|1800|1386|85
```

## How to use this

Review the cluster grids instead of the original recordings one by one:

```text
clusters/cluster_00_grid.png
clusters/cluster_01_grid.png
...
```

Then assign semantic labels to clusters, for example:

```text
clean_low_primary
rising_chip
broad_upper_sweep
background_or_bad_box
uncertain
```

After cluster labels are stable, use them to define cluster-specific pseudo-box rules and filter out noisy candidates.
