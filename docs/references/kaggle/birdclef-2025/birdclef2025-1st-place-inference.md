---
slug: "nikitababich/birdclef2025-1st-place-inference"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 65
---

# Kaggle Summary: BirdCLEF 2025 1st Place Solution

**Source**: https://www.kaggle.com/code/nikitababich/birdclef2025-1st-place-inference
**Competition**: BirdCLEF 2025
**Score**: 1st place (Private LB score not disclosed in inference notebook)

## Summary

The winning solution for BirdCLEF 2025 uses a multi-model ensemble combining seven different models: five self-supervised learning models trained with pseudo-labeling and two supervised baseline models. The approach employs Sound Event Detection with attention-based temporal pooling, OpenVINO-accelerated inference, and sophisticated post-processing including Gaussian temporal smoothing and weighted ensemble averaging.

## Approach

### Architecture

The core architecture is a Sound Event Detection (SED) model with three main components:

**Backbone**: Multiple CNN architectures from the timm library, including:
- EfficientNet variants (B0, B3, B4)
- RegNetY (008, 016)
- ECA-NFNetL0

All backbones use ImageNet pre-trained weights from timm, configured as feature extractors with `features_only=True`.

**Head**: Custom attention-pooling head (`AttHead`) that:
1. Applies GeM (Generalized Mean) pooling across frequency dimension only, preserving temporal structure
2. Projects through dense layers with dropout (512 hidden units, 50% dropout)
3. Computes framewise logits via 1x1 convolution per species class
4. Returns per-frame predictions rather than clip-level aggregation

**Spectrogram Generation**: On-the-fly mel-spectrogram computation using torchaudio transforms:
- 224 mel bins, 512 time frames (20 seconds duration)
- FFT size: 4096, hop length: 1252 samples (32kHz sample rate)
- Frequency range: 0-16000 Hz
- Normalization: Custom "default" scheme that standardizes then min-max scales to [0,1]
- AmplitudeToDB with 80dB top_db parameter

### Data Pipeline

**Audio Loading**: Parallel loading of entire test soundscapes using librosa with kaiser_best resampling to 32kHz.

**Segment Extraction**: Two strategies implemented (solution uses "full signal" approach):
1. Full-signal method: Extract one long spectrogram, then slice into overlapping windows to preserve spectral continuity
2. Segment method: Split waveform first, compute spectrograms independently

**Windowing**: 20-second spectrograms with 5-second step size (75% overlap), yielding 12 segments per 60-second soundscape. Padding applied to handle edges.

**Augmentation**: No test-time augmentation applied during inference (training augmentations included mixup, likely drop path based on config references).

### Training

Training details are not fully disclosed in the inference notebook, but configuration reveals:

**Loss Function**: Cross-entropy loss, likely with dual supervision on both framewise and clipwise predictions (0.5 weight each based on commented code).

**Models Breakdown**:
- 5 self-learning models: Iterative pseudo-labeling (iterations 3-4), temperature scaling (0.55-0.6), mixup (ratio 0.5-1.0)
- 2 supervised models: One standard training, one specialized on insects/amphibia classes
- Batch sizes: 54-128 depending on model size
- Drop path rate: 0.15
- Epochs: 15-40 depending on configuration

**Multi-Label Strategy**: One model (EfficientNet-B0) was trained on an expanded 700-class taxonomy including insects, amphibians, and mammals, then mapped back to the 206 BirdCLEF classes using a class mapping dictionary. This captures non-bird sounds that may appear in soundscapes.

### Inference / Post-Processing

**Overlap Reconstruction**: Custom algorithm (`overlap_average_max_delta`) that:
1. Reconstructs full framewise predictions by averaging overlapping regions
2. Within each 5-second segment, takes max across frames
3. Applies temporal delta offsets (+/- 2 frames) and blends: 50% center + 25% left-shifted + 25% right-shifted
4. This creates temporal smoothness while preserving peak activations

**Ensemble Strategy**:
- 7 models weighted by learned coefficients: [0.133, 0.166, 0.133, 0.133, 0.166, 0.133, 0.133]
- Power transformation with exponent 1 (no actual power adjustment in winning config)
- Weighted averaging of predicted probabilities

**Temporal Smoothing**:
- Gaussian convolution with 5-frame kernel [0.1, 0.2, 0.4, 0.2, 0.1]
- Applied along temporal axis after ensemble to reduce noise

**Optimization**:
- OpenVINO conversion of backbone models for CPU inference acceleration
- Pre-conversion to INT8/FP16 precision (XML format)
- Parallel audio loading using joblib
- Head remains in PyTorch FP16 autocast

## Key Ideas

- **Framewise SED instead of clip classification**: The model predicts per-frame probabilities and carefully reconstructs them with overlap handling, rather than directly predicting clip-level presence. This better handles partial detections within segments.

- **Delta-shifted temporal aggregation**: The `overlap_average_max_delta` method blends predictions from three temporal alignments (center, +2 frames, -2 frames) weighted 0.5/0.25/0.25. This acts as temporal test-time augmentation without multiple forward passes.

- **Multi-taxonomy training**: Training one model on 700 classes (birds + insects + amphibians + mammals) and projecting to the target 206 classes allows the model to learn richer representations and better handle non-target sounds common in soundscapes.

- **Self-supervised pseudo-labeling pipeline**: Five of seven models used iterative pseudo-labeling (3-4 iterations) with temperature-scaled predictions (0.55-0.6) to leverage unlabeled data, suggesting significant performance gains from semi-supervised learning.

- **GeM pooling only across frequency**: Unlike standard GeM that pools spatially, this implementation pools only the frequency axis (224 mel bins → 1) while preserving all time steps, maintaining temporal resolution for SED.

- **OpenVINO hybrid inference**: Only the heavy backbone is converted to OpenVINO (INT8/FP16), while the lightweight head stays in PyTorch FP16. This balances speed and flexibility.

- **Full-signal spectrogram extraction**: Computing one long mel-spectrogram and slicing it maintains phase coherence across segment boundaries, versus computing independent spectrograms per segment.

## Results

| Metric | Value |
|--------|-------|
| CV     | Not disclosed |
| Public LB | Not disclosed |
| Private LB | 1st place (exact score not in inference notebook) |
| Inference time | ~13 seconds per soundscape (estimated from notebook structure) |

## Connections

- **Related concepts**:
  - Sound Event Detection (SED) for bioacoustics
  - Pseudo-labeling / self-supervised learning
  - Temporal aggregation strategies
  - GeM pooling variants

- **Informs experiments**:
  - Experiment 008 explored attention-pooled SED but used mean pooling; this solution uses GeM-frequency pooling
  - Experiment 009 uses clip-level classification; this demonstrates framewise SED advantages
  - Multi-iteration pseudo-labeling could inform future semi-supervised experiments

- **Similar approaches**:
  - Framewise SED common in audio tagging competitions (AudioSet, DCASE)
  - Pseudo-labeling iterations similar to noisy student training
  - Multi-taxonomy training relates to hierarchical classification approaches
