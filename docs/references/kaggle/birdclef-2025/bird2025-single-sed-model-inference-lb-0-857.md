---
slug: "i2nfinit3y/bird2025-single-sed-model-inference-lb-0-857"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 329
---

# Kaggle Summary: Single SED Model Inference

**Source**: https://www.kaggle.com/code/i2nfinit3y/bird2025-single-sed-model-inference-lb-0-857
**Competition**: BirdCLEF 2025
**Score**: 0.857 (Public LB)

## Summary

This notebook implements Sound Event Detection (SED) inference for BirdCLEF 2025 using a single seresnext26t_32x4d model with attention-based pooling. The approach processes soundscape audio in overlapping 10-second windows centered on 5-second target segments, applies temporal smoothing across adjacent predictions, and uses test-time augmentation to improve robustness.

## Approach

### Architecture

The model uses a **seresnext26t_32x4d** backbone from the timm library without ImageNet pretraining. Audio is converted to mel spectrograms on-the-fly using torchaudio transforms with the following parameters:
- Sample rate: 32kHz
- Mel bins: configuration-dependent (stored in checkpoint)
- Normalization: amplitude-to-dB conversion with 80 dB dynamic range scaling

The architecture consists of:
1. Batch normalization applied to the mel spectrogram frequency axis
2. CNN backbone encoder (all layers except final classifier)
3. Frequency-axis average pooling to create a time-frequency representation
4. Channel smoothing via combined max and average pooling along the time axis
5. Fully connected projection layer with ReLU activation and dropout (0.5)
6. **AttBlockV2** attention mechanism that produces both clipwise and framewise predictions

The attention block uses two parallel 1D convolutions: one for attention weights (with tanh activation and softmax normalization) and one for class activations (with sigmoid). During inference, framewise predictions are extracted from a centered temporal window and aggregated using max pooling.

### Data Pipeline

Audio loading strategy for inference:
- Each soundscape file is processed in 5-second segments
- Audio is **triple-concatenated** to enable circular padding at boundaries
- For each 5-second target segment, a **10-second window** is extracted, centered on the target
- Padding is zeroed at file boundaries (first and last segments)
- Audio is loaded with soundfile (librosa backend) and kept at original sample rate

Mel spectrogram computation happens inside the model via torchaudio transforms, ensuring consistency between training and inference. The spectrogram normalization divides by 80 after adding 80 (shifting range to [0,1]).

### Training

Training details are not included in this inference-only notebook, but references point to a separate training repository (birdclef2023-2nd-place-solution by LIHANG-HONG). The model checkpoint includes the full configuration used during training.

Key training configuration stored in checkpoint:
- Loss: Binary cross-entropy (implied by "train_bce" stage)
- Dropout: 0.5 applied before and after the FC projection
- Drop path: 0.2 in backbone
- Backbone drop rate: 0.2

### Inference / Post-Processing

**Test-Time Augmentation (TTA)**:
- Three temporal shifts are applied during attention pooling: centered, -2 frames, +2 frames
- Predictions are ensemble-weighted: 0.5 × center + 0.25 × left shift + 0.25 × right shift
- The temporal window for attention pooling is calculated based on the ratio of inference duration (5s) to training duration (10s)

**Temporal Smoothing**:
After generating predictions for all segments, the submission undergoes temporal smoothing within each soundscape file:
- First segment: 0.8 × current + 0.2 × next
- Last segment: 0.8 × current + 0.2 × previous
- Middle segments: 0.2 × previous + 0.6 × current + 0.2 × next

This enforces temporal consistency based on the assumption that bird presence changes gradually across adjacent 5-second windows.

**Parallel Processing**:
Multiple soundscape files are processed concurrently using ThreadPoolExecutor with 4 workers.

## Key Ideas

- **Centered window extraction**: Using 10-second audio windows centered on each 5-second target segment provides temporal context that improves predictions compared to using isolated 5-second clips
- **Triple concatenation trick**: Concatenating the audio three times enables seamless extraction of centered windows even at file boundaries without special-case logic
- **Framewise max pooling for inference**: Instead of weighted attention pooling (which is used during training), inference extracts the maximum framewise prediction across the temporal window—this appears more robust for detecting brief bird calls
- **TTA via temporal jitter**: Small frame-level shifts (±2 frames) during attention pooling provide regularization without requiring multiple mel spectrogram computations
- **Post-hoc temporal smoothing**: The weighted averaging across adjacent predictions (0.2/0.6/0.2 kernel) acts as a low-pass filter that reduces spurious isolated predictions
- **On-device spectrogram computation**: Using torchaudio transforms within the model eliminates precomputation overhead and ensures exact train/test consistency

## Results

| Metric | Value |
|--------|-------|
| CV     | Not reported |
| Public LB | 0.857 |
| Private LB | Not reported |

The 329 votes indicate this was a popular and effective solution during the competition.

## Connections

- **Related concepts**:
  - Sound Event Detection (SED) with attention pooling
  - Temporal context windows for audio classification
  - Test-time augmentation via temporal shifts
- **Informs experiments**:
  - Potential application of AttBlockV2 attention mechanism to SSAMBA or other temporal models
  - Temporal smoothing as post-processing for soundscape predictions
  - Comparison of attention-weighted vs. max-pooled framewise predictions
- **Similar approaches**:
  - Training methodology references 2023 BirdCLEF 2nd place solution architecture
  - Inference pipeline structure similar to efficientnet-b0-pytorch-inference-birdclef-25
