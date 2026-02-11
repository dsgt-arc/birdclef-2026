---
name: "bambird + scikit-maad"
source: "https://github.com/ear-team/bambird"
license: "BSD-3-Clause (both)"
category: "utility"
pipeline_stage:
  - "data-processing"
relevance: "Unsupervised bird vocalization event detection used by BirdSet for detected_events annotations"
status: "reference"
---

# bambird + scikit-maad

## Overview

**bambird** is an unsupervised pipeline for detecting and classifying bird vocalizations in audio recordings. It builds on **scikit-maad** (Modular Audio Analysis in Python), a signal-processing library for soundscape ecology. Together they provide a lightweight, CPU-only alternative to neural detectors like BirdNET for identifying when birds are vocalizing.

BirdSet uses bambird to generate `detected_events` timestamps on XenoCanto focal recordings (see [BirdSet paper, Section 3](https://arxiv.org/abs/2403.10380)).

## Pipeline

bambird operates in three stages:

### 1. Time-Frequency ROI Extraction (scikit-maad)

Extracts Regions of Interest (bounding boxes in the spectrogram) using signal processing:

```python
import maad

# Load and compute spectrogram
s, fs = maad.sound.load("recording.wav")
Sxx, tn, fn, ext = maad.sound.spectrogram(s, fs)

# Remove background noise
Sxx_noNoise = maad.sound.median_equalizer(Sxx)

# Create binary mask via energy thresholding
im_bin = maad.rois.create_mask(Sxx_noNoise, bin_h=0.7, bin_l=0.2)

# Extract ROIs as bounding boxes
im_rois, df_rois = maad.rois.select_rois(im_bin, min_roi=25)
```

Two extraction modes:
- `extract_rois_core`: most energetic parts only (tighter boxes)
- `extract_rois_full_sig`: full call extent (wider temporal coverage)

### 2. Feature Computation

For each ROI, computes acoustic features:
- **Shape features** (26 dimensions): bidimensional wavelet coefficients capturing spectro-temporal texture
- **Centroid features**: spectral centroid and bandwidth

```python
# Compute shape features for each ROI
df_shape, params = maad.features.shape_features(
    Sxx_noNoise, resolution='low', rois=df_rois
)
```

### 3. DBSCAN Clustering

Clusters ROIs by acoustic similarity. The dominant cluster is assumed to be the target species (since XenoCanto recordings are focal — one primary species):

```python
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

# Reduce dimensions
X_tsne = TSNE(n_components=2).fit_transform(features)

# Cluster
clusters = DBSCAN(eps=0.5, min_samples=3).fit_predict(X_tsne)
# cluster == -1 → noise/outlier
# largest cluster → target species vocalizations
```

## Output Format

BirdSet stores bambird output as HuggingFace dataset columns:

| Field | Type | Description |
|-------|------|-------------|
| `detected_events` | `Sequence(Sequence(float64))` | List of `[start_s, end_s]` pairs |
| `event_cluster` | `Sequence(int64)` | DBSCAN cluster ID per event (-1 = noise) |
| `peaks` | `Sequence(float64)` | Peak timestamps from scipy peak detection |

Example:
```python
{
    'detected_events': [[0.736, 1.824], [9.936, 10.944], [13.872, 15.552]],
    'event_cluster': [0, 0, 0],
    'peaks': [1.28, 10.44, 14.71]
}
```

No confidence scores — DBSCAN gives hard cluster assignments.

## How BirdSet Uses Events

BirdSet's training pipeline has two components that consume `detected_events`:

1. **`XCEventMapping`** (`birdset/datamodule/components/event_mapping.py`):
   - Expands each recording into multiple samples, one per detected event
   - Filters noise clusters (cluster = -1) by default
   - `biggest_cluster=True` keeps only the dominant cluster

2. **`EventDecoding`** (`birdset/datamodule/components/event_decoding.py`):
   - Extracts audio segments centered on each event
   - Pads short events to `extension_time`
   - Random offset within the padded window for augmentation

## Comparison with BirdNET

| | bambird | BirdNET |
|---|---------|---------|
| **Approach** | Signal processing + unsupervised clustering | Neural network (CNN) |
| **Output** | Event timestamps + cluster IDs | Species predictions + confidence scores |
| **Species ID** | No (just "dominant sound" vs noise) | Yes (6500+ species) |
| **Confidence** | No (hard cluster assignment) | Yes (0-1 per species) |
| **Dependencies** | scikit-maad, sklearn, scipy | TensorFlow/TFLite |
| **Compute** | CPU-only, lightweight | CPU or GPU |
| **License** | BSD-3-Clause | MIT code, CC BY-NC-SA 4.0 models |
| **Best for** | Detecting *when* vocalizations occur | Detecting *what species* is vocalizing |

## Dependencies

- **scikit-maad** (`maad`): [GitHub](https://github.com/scikit-maad/scikit-maad), [Docs](https://scikit-maad.github.io/)
- **bambird**: [GitHub](https://github.com/ear-team/bambird)
- scikit-learn (DBSCAN, t-SNE)
- scipy (peak detection)
- NumPy, matplotlib

Both are pip-installable:
```bash
pip install scikit-maad bambird
```

## References

- Michaud, F., Sueur, J., Le Cesne, M., & Haupert, S. (2022). "Unsupervised classification to improve the quality of a bird song recording dataset." *Ecological Informatics*, 74, 101952. [DOI](https://doi.org/10.1016/j.ecoinf.2022.101952)
- Ulloa, J.S., Haupert, S., Latorre, J.F., Aubin, T., & Sueur, J. (2021). "scikit-maad: An open-source and modular toolbox for quantitative soundscape ecology in Python." *Methods in Ecology and Evolution*, 12(12), 2334-2340.
- [scikit-maad tutorial: Unsupervised sound classification](https://scikit-maad.github.io/_auto_examples/2_advanced/plot_unsupervised_sound_classification.html)
