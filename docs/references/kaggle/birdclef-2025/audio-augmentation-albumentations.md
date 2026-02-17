---
slug: "gopidurgaprasad/audio-augmentation-albumentations"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 87
---

# Kaggle Summary: Audio Augmentation - Albumentations

**Source**: https://www.kaggle.com/code/gopidurgaprasad/audio-augmentation-albumentations
**Competition**: BirdCLEF 2025
**Score**: Not applicable (tutorial/library notebook)

## Summary

This notebook demonstrates how to build a composable audio augmentation pipeline using the Albumentations library framework, adapted for audio tasks. The author creates custom transform classes that wrap common audio augmentation techniques, enabling them to be chained together in a declarative pipeline similar to how Albumentations handles image augmentations.

## Approach

### Architecture

Not applicable - this is a data augmentation tutorial, not a model training notebook.

### Data Pipeline

The notebook demonstrates a complete audio preprocessing and augmentation pipeline:

1. **Audio loading**: Loads MP3 files using pydub, converts to numpy arrays at 16kHz sample rate
2. **Base transform class**: Implements `AudioTransform` that extends Albumentations' `BasicTransform` to work with audio tuples (waveform, sample_rate)
3. **Augmentation techniques** implemented as composable transforms:
   - **NoiseInjection**: Adds Gaussian noise scaled by a random factor (0 to 0.5)
   - **ShiftingTime**: Circularly shifts the waveform left or right, filling edges with silence
   - **PitchShift**: Changes pitch by random semitone steps (wraps librosa function)
   - **TimeStretch**: Stretches or compresses the audio timeline by a random rate
   - **RandomAudio**: Randomly crops or pads audio to fixed duration (5 seconds)
   - **MelSpectrogram**: Converts waveform to mel-scaled power spectrogram in dB scale
   - **SpecAugment**: Applies frequency and time masking to spectrograms (random rectangular masks)
   - **SpectToImage**: Creates 3-channel image from spectrogram using original + delta + delta-delta features

4. **Pipeline composition**: Uses `albumentations.Compose()` to chain transforms with individual probability controls

### Training

Not applicable - no model training in this notebook.

### Inference / Post-Processing

Not applicable - focuses on data augmentation only.

## Key Ideas

- **Albumentations pattern for audio**: By extending `BasicTransform` and defining a `targets` property, audio augmentations can use the same declarative API as image augmentations, enabling probability-based composition and reproducible pipelines.

- **Tuple-based data format**: Maintains (waveform, sample_rate) as a paired tuple throughout the pipeline, allowing transforms to access sample rate when needed (e.g., for librosa functions).

- **Spectrogram as intermediate representation**: Pipeline progresses from raw waveform → waveform augmentations → spectrogram → spectrogram augmentations → multi-channel image, separating time-domain and frequency-domain operations.

- **Delta features for 3-channel input**: Stacks mel spectrogram with its first and second derivatives to create a 3-channel representation compatible with standard image models (similar to how RGB images work).

- **SpecAugment masking strategy**: Masks random frequency bands and time segments on spectrograms, with configurable masking percentage (default 15% frequency, 20% time) applied 1-2 times per sample.

- **Time shifting with silence padding**: When circularly shifting audio, the technique fills wrapped-around portions with zeros rather than letting them loop, preventing unnatural discontinuities.

## Results

Not applicable - this is an educational/library notebook demonstrating techniques rather than reporting competition scores.

## Connections

- **Related concepts**: This approach directly relates to our on-the-fly audio pipeline in experiment 008, which uses nnAudio instead of librosa for GPU-accelerated spectrogram computation
- **Informs experiments**: The delta+delta-delta feature stacking could be tested as an alternative to our single-channel mel spectrograms in future experiments
- **Similar approaches**: The SpecAugment implementation is similar to standard approaches but operates on mel spectrograms rather than raw spectrograms
