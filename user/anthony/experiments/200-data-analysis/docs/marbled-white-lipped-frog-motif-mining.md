# Notebook: Marbled White-lipped Frog Motif Mining

## Header

| Field | Value |
|---|---|
| Date | 2026-05-10 |
| Status | Simplified current pipeline |
| BirdCLEF label | `22967` |
| Species | Marbled White-lipped Frog / *Leptodactylus elenae* |
| Mining code | `src/data_analysis/marbled_frog/motif_mining.py` |
| Bank code | `src/data_analysis/marbled_frog/motif_bank.py` |
| Current artifacts | `artifacts/marbled_frog/motif_mining_simplified/` |

## Goal

Discover recurring time-frequency motif families in the `22967` train audio and build a small, reviewable motif bank. The pipeline should be defensible and library-backed rather than a stack of custom heuristics.

## Current simplified pipeline

```text
audio
  → robust-excess-z spectrogram for detection/visualization
  → scikit-maad create_mask hysteresis ROI proposal
  → scikit-maad select_rois bounding boxes
  → scikit-maad shape_features + centroid_features
  → simple edge-touch marking
  → StandardScaler → sklearn HDBSCAN
  → cluster review grids
  → motif bank: feature-space medoid + nearest neighbors
  → fixed physical support per motif family for templates
```

Current artifact root:

```text
artifacts/marbled_frog/motif_mining_simplified/
```

Key outputs:

```text
tables/candidates.csv
tables/feature_matrix.csv
tables/cluster_summary.csv
plots/candidate_feature_pca.png
clusters/cluster_XX_grid.png
motif_bank/plots/cluster_XX_motif_bank.png
motif_bank/tables/motif_bank_clusters.csv
motif_bank/tables/motif_bank_members.csv
motif_bank/templates/cluster_XX_templates.npz
```

## Design decisions

### Detector ROI vs template support

Raw maad ROI boxes are **detection/provenance boxes** only. They are allowed to vary in physical area because thresholded connected regions naturally vary with SNR, tails, and fragmentation.

Motif-bank templates use a separate **canonical physical support** per family:

```text
time window: fixed pre/post duration around each candidate peak_time_sec
frequency band: fixed fmin/fmax estimated from robust ROI quantiles in the family
```

In motif-bank plots:

```text
cyan rectangle       = fixed family template support
dashed green rectangle = raw maad ROI provenance box
```

This avoids comparing a tiny edge fragment and a large full call as if they were the same physical object after resizing.

### Removed exploratory machinery

The following exploratory scripts were committed for history and then removed from the main pipeline:

```text
src/data_analysis/marbled_frog/motif_cluster_outliers.py
src/data_analysis/marbled_frog/motif_spectral_prior.py
```

Rationale: pairwise custom template distances, recording-aware spectral priors, and weighted cluster-validity scores introduced too many hand-tuned thresholds. The simplified pipeline instead relies on scikit-maad features and HDBSCAN, with human review of clusters.

## Current run summary

The simplified run detected 270 candidates. HDBSCAN produced 9 non-noise clusters plus a noise/edge/unclustered label `-1`.

```text
cluster -1: 104 candidates, 8 recordings, mostly edge/noise/unclustered
cluster  0:   9 candidates, 1 recording
cluster  1:  13 candidates, 1 recording
cluster  2:  11 candidates, 2 recordings
cluster  3:  15 candidates, 1 recording
cluster  4:   8 candidates, 1 recording
cluster  5:   9 candidates, 1 recording
cluster  6:  41 candidates, 4 recordings
cluster  7:   6 candidates, 2 recordings
cluster  8:  54 candidates, 2 recordings
```

The current motif bank is generated for all non-noise clusters as **review candidates**, not final accepted biological families. Review plots live at:

```text
artifacts/marbled_frog/motif_mining_simplified/motif_bank/plots/
```

## Related synthesis

The current end-to-end synthesis, empirical findings, and load-bearing assumptions are documented in:

```text
docs/marbled-white-lipped-frog-motif-synthesis.md
```

## Next review step

Open the simplified cluster grids and motif-bank plots. For each HDBSCAN cluster, mark one of:

```text
accept as motif family
reject as artifact/noise
split needed / mixed family
```

After human approval, rerun `motif_bank.py --approved-clusters ...` so the bank contains only accepted families.
