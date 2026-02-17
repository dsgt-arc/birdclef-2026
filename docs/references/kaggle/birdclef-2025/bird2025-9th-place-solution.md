---
slug: "i2nfinit3y/bird2025-9th-place-solution"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 17
---

# Kaggle Summary: Bird2025 9th Place Solution

**Source**: https://www.kaggle.com/code/i2nfinit3y/bird2025-9th-place-solution
**Competition**: BirdCLEF 2025
**Score**: 9th place (exact score not disclosed in notebook)

## Summary

This solution achieves 9th place through an ensemble of three SEResNeXt26t models with attention-based pooling, temporal smoothing, and test-time augmentation. The key differentiator is sophisticated temporal consistency enforcement through both center-crop TTA and post-processing smoothing across consecutive 5-second segments within each soundscape.

## Approach

### Architecture

The model uses a custom Sound Event Detection (SED) architecture with the following components:

- **Backbone**: SEResNeXt26t (squeeze-excitation ResNeXt with 32x4d configuration) from timm library
- **Pooling**: Attention-based pooling module (AttBlockV2) that learns both attention weights and classification embeddings per time frame
- **Inference strategy**: Maximum framewise prediction within a centered temporal window, rather than attention-weighted averaging
- **Pre-training**: Not specified; models appear trained from scratch or with ImageNet initialization

The attention module uses two parallel 1D convolutions: one learns soft attention weights (via softmax over tanh), the other learns classification features (with sigmoid activation). During inference, the model extracts maximum predictions across frames within a target duration window.

### Data Pipeline

**Spectrogram generation**:
- Sample rate: 32 kHz
- Mel filterbank: 128 bins (configurable via n_mels), HTK scaling
- Frequency range: f_min to (f_max - 2000 Hz)
- Normalization: Min-max scaling per sample after dB conversion, divided by 80 dB range
- Input duration: 10 seconds of audio for training, 5 seconds for inference targets

**Audio preprocessing**:
- Audio files are concatenated three times (original + 2 repeats) to enable padding handling
- Each 5-second target segment is extracted from a 10-second window with 2.5-second padding on each side
- Edge segments have their padding zeroed out to avoid boundary artifacts
- Multiple normalization schemes supported: 80 dB scaling, min-max, z-score clipping to [-3, 3]

**Augmentation**: Not explicitly shown in inference code, but the model architecture includes a "spec_augmenter" placeholder in training mode.

### Training

Training details are not included in this inference-only notebook. Configuration hints from checkpoint loading:

- **Loss**: Appears to be BCE-based (stage named 'train_bce')
- **Dropout**: 0.5 applied before and after the FC layer during feature extraction
- **Regularization**: Batch normalization on mel bins, channel smoothing via max+avg pooling
- **Hardware**: Models are exported to ONNX and run on CPU during inference

### Inference / Post-Processing

**Ensemble**: Three models averaged with weights [0.4, 0.4, 0.2]
- Two EfficientNetV2-B3 variants (different seeds)
- One SEResNeXt26t model

**Test-time augmentation**: Three temporal crops per prediction
- Center crop aligned to 5-second target (50% weight)
- Crop shifted 2 frames earlier (25% weight)
- Crop shifted 2 frames later (25% weight)
- Frame offsets are in feature space, not raw audio time

**Temporal smoothing**: Two-stage smoothing strategy
1. **Per-model smoothing**: Applied before ensemble, smooths predictions across consecutive 5-second segments within each soundscape
2. **Post-ensemble smoothing** (for single models only): Additional smoothing after final predictions
   - First segment: 80% current + 20% next
   - Last segment: 80% current + 20% previous
   - Middle segments: 20% previous + 60% current + 20% next

**Output processing**: Framewise maximum across the target temporal window, rather than attention-weighted pooling used during training.

## Key Ideas

- **Attention inference divergence**: The model trains with attention-weighted pooling but infers using framewise maximum within a temporal window—this decoupling allows the attention mechanism to learn good temporal localization while inference maximizes sensitivity to brief vocalizations.

- **Multi-scale temporal smoothing**: Applying smoothing both per-model (before ensemble) and optionally after ensemble ensures temporal consistency at multiple abstraction levels, reducing spurious single-segment predictions.

- **Centered TTA with asymmetric weighting**: Rather than uniform averaging, the center crop receives 50% weight while early/late crops get 25% each—this balances temporal robustness with preference for the annotated center alignment.

- **Audio padding strategy**: The triple-concatenation trick enables proper edge handling by creating a circular buffer, then zeroing out the padded regions for true edge segments—this avoids artifacts from reflecting or zero-padding alone.

- **ONNX deployment on CPU**: All three models are exported to ONNX and run with ONNXRuntime, making the solution hardware-agnostic and efficient for CPU-only inference environments.

- **Channel smoothing**: After backbone feature extraction, applying both max pooling and average pooling with kernel size 3 along the time dimension (then summing) acts as a regularizer that encourages smooth temporal activations.

## Results

| Metric | Value |
|--------|-------|
| CV     | Not reported |
| Public LB | Not reported |
| Private LB | 9th place |

## Connections

- **Related concepts**:
  - Temporal consistency in soundscape prediction
  - Attention-based pooling for variable-length sequences
  - Test-time augmentation in audio domain

- **Informs experiments**:
  - Could inform experiment design for temporal smoothing post-processing
  - Attention pooling mechanism relevant to experiment 008 (which tested SED attention pooling)
  - Multi-model ensemble strategies with weighted averaging

- **Similar approaches**:
  - Attention-pooling SED architecture similar to PANNs (Kong et al.)
  - Temporal smoothing strategy common in audio event detection tasks
  - Center-crop TTA conceptually related to multi-crop evaluation in image classification
