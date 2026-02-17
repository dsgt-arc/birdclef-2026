---
slug: "myso1987/post-processing-with-power-adjustment-for-low-rank"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 351
---

# Kaggle Summary: Post-Processing with Power Adjustment for Low-Rank

**Source**: https://www.kaggle.com/code/myso1987/post-processing-with-power-adjustment-for-low-rank
**Competition**: BirdCLEF 2025
**Score**: Unknown

## Summary

This notebook introduces a post-processing technique that selectively suppresses predictions for low-confidence species classes by applying power transformations. The author notes that while this improves leaderboard scores, it may overfit to the test distribution. The approach is applied at inference time after model prediction, requiring no retraining.

## Approach

### Architecture

The notebook uses an ECA-NFNet-L0 backbone with a Sound Event Detection (SED) head:
- Timm-based encoder (ECA-NFNet-L0) with ImageNet-style pretraining removed
- Attention-based pooling module (AttBlockV2) that computes softmax over time steps
- Frame-wise predictions aggregated via attention-weighted sum
- Ensemble of 3 models with identical architecture

### Data Pipeline

Audio preprocessing follows standard BirdCLEF conventions:
- 5-second chunks extracted from 60-second test soundscapes (12 chunks per file)
- 32 kHz sample rate
- Mel spectrograms: 128 bins, 1024 FFT size, 512 hop length, 50-16000 Hz frequency range
- Slaney normalization with HTK mel scale
- Log transformation followed by standardization (mean/std normalization per spectrogram)
- Mono audio conversion from stereo input

### Training

Training details not included in this inference-only notebook. The notebook loads pre-trained model weights from a separate dataset (birdclef-2025-sed-models-p).

### Inference / Post-Processing

The core innovation is a two-stage post-processing pipeline:

1. **Power adjustment for low-ranked classes**: For each chunk prediction, species are ranked by their maximum probability across all chunks in the soundscape. Species ranked below top-K (default K=30) have their probabilities raised to a power (default exponent=2, i.e., squared). This suppresses low-confidence predictions while preserving high-confidence ones.

2. **Temporal smoothing**: After ensemble averaging, predictions are smoothed across adjacent 5-second chunks using weighted averaging. Middle chunks use a 0.2-0.6-0.2 kernel (20% previous chunk, 60% current, 20% next). Edge chunks use 0.8-0.2 weighting to avoid boundary artifacts.

The power adjustment is applied per-model before ensemble averaging. Ensemble uses simple arithmetic mean across the 3 models.

## Key Ideas

- **Selective suppression by rank**: Rather than applying a global threshold or scaling all classes uniformly, the power transformation targets only species with low maximum probability across the soundscape. This exploits the intuition that truly present species should have at least one strong detection across the 60-second recording.

- **Per-chunk ranking**: The ranking is computed based on column-wise maximum (i.e., the strongest detection for each species across all 12 chunks), not row-wise. This means a species that appears strongly in even one chunk will be preserved in all chunks.

- **Squaring as confidence sharpening**: The exponent=2 choice creates a nonlinear gap between moderate probabilities (e.g., 0.3 → 0.09) and high probabilities (0.9 → 0.81). This acts as a soft threshold that disproportionately penalizes uncertain predictions.

- **Two-scale smoothing**: Temporal smoothing operates at the chunk level (5-second scale) rather than frame level, providing robustness to transient predictions without losing temporal resolution needed for the submission format.

- **Overfitting warning**: The author explicitly acknowledges this may overfit to the public leaderboard distribution, suggesting the top-K and exponent values were likely tuned on test-time feedback rather than cross-validation.

## Results

No cross-validation or leaderboard scores reported in the notebook.

## Connections

- **Related concepts**: This technique addresses the class imbalance and long-tail distribution problem in BirdCLEF, where many species have very few training examples and may produce false positives at test time.
- **Informs experiments**: The power adjustment technique could be combined with other post-processing strategies like species co-occurrence filtering or temporal consistency constraints.
- **Similar approaches**: Temporal smoothing is a common technique in SED tasks; see other BirdCLEF 2025 submissions for variations on chunk aggregation strategies.
