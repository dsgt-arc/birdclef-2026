---
slug: "kadircandrisolu/transforming-audio-to-mel-spec-birdclef-25"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 450
---

# Kaggle Summary: Transforming Audio-to-Mel Spec. | BirdCLEF'25

**Source**: https://www.kaggle.com/code/kadircandrisolu/transforming-audio-to-mel-spec-birdclef-25
**Competition**: BirdCLEF 2025
**Score**: N/A (preprocessing notebook)
**Related Dataset**: [BirdCLEF'25 | Mel Spectrograms](https://www.kaggle.com/datasets/kadircandrisolu/birdclef25-mel-spectrograms)

## Summary

This notebook demonstrates a preprocessing pipeline for converting raw BirdCLEF audio files into normalized mel-spectrogram images suitable for training 2D convolutional neural networks. The author provides both a debug mode for quick testing and a full pipeline that processes all training data, outputting a standardized dataset of mel-spectrogram representations.

## Approach

### Data Pipeline

The preprocessing workflow transforms audio recordings into fixed-size mel-spectrogram images through several stages:

1. **Audio Loading**: Audio files are loaded at 32 kHz sampling rate using librosa
2. **Temporal Standardization**: All recordings are standardized to 5-second duration by:
   - Extracting the center portion of recordings longer than 5 seconds
   - Repeating (tiling) recordings shorter than 5 seconds multiple times
   - Zero-padding any remaining gap to reach exactly 5 seconds
3. **Mel-Spectrogram Conversion**: Audio is transformed using the following parameters:
   - FFT window size: 1024
   - Hop length: 512 samples
   - Number of mel bins: 128
   - Frequency range: 50 Hz to 14 kHz
   - Power spectrum squared (power=2.0)
4. **Normalization**: The power spectrogram is converted to decibel scale, then min-max normalized to [0, 1] range
5. **Image Resizing**: Final spectrograms are resized to 256x256 pixels using bilinear interpolation
6. **NaN Handling**: Any NaN values in the audio signal are replaced with the mean of the valid samples

The pipeline handles edge cases like missing data and variable-length recordings to ensure consistent output dimensions across all samples.

### Metadata Processing

The notebook builds a working dataframe that maps:
- Species labels to numeric class IDs
- Filenames to full filepaths
- Species to their taxonomic class (Aves, Amphibia, Insecta, Mammalia)
- Creates unique sample identifiers combining directory and filename

### Storage Format

Processed mel-spectrograms are stored as float32 numpy arrays in a dictionary keyed by sample name, intended for serialization to disk in the full processing run.

## Key Ideas

- **Center-cropping strategy**: Extracting the central 5-second segment assumes bird vocalizations are most likely in the middle of recordings, avoiding edge artifacts
- **Frequency range tuning**: The 50-14000 Hz range captures relevant bioacoustic signals while filtering out irrelevant low-frequency noise and exceeding typical avian vocalization ranges
- **Fixed aspect ratio**: 256x256 output enables direct use with standard computer vision architectures designed for square images
- **Power spectrum**: Using power=2.0 creates a power spectrogram rather than magnitude, potentially emphasizing stronger signals
- **Debug mode design**: Processing only 50 samples with one per class allows quick pipeline validation before committing to full preprocessing
- **Decibel normalization**: Converting to dB scale with max reference provides perceptually-relevant amplitude representation before min-max scaling

## Results

This is a preprocessing notebook, not a modeling solution, so no competition scores are applicable. The output is a preprocessed dataset published separately on Kaggle.

| Metric | Value |
|--------|-------|
| Input format | OGG audio files |
| Output format | 256x256 float32 mel-spectrograms |
| Processing speed | ~50 files processed in debug mode |
| Target duration | 5.0 seconds |
| Sample rate | 32000 Hz |

## Connections

- **Related concepts**: This preprocessing approach is complementary to our on-the-fly audio pipeline used in experiments 008-009, which computes mel-spectrograms during training rather than pre-computing them
- **Key differences from our approach**:
  - This uses librosa for CPU-based preprocessing vs. our nnAudio GPU pipeline
  - Fixed 256x256 output vs. our variable rectangular patches (128x1, 16x16)
  - Center-crop only vs. our random crop augmentation during training
  - 50-14kHz range vs. our broader frequency coverage
- **Potential applications**: The pre-computed dataset approach could reduce training-time I/O if disk space is available, trading storage for compute
