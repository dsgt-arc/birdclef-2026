---
slug: "sietze535/pseudo-label-creator"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 29
---

# Kaggle Summary: Pseudo-label Creator

**Source**: https://www.kaggle.com/code/sietze535/pseudo-label-creator
**Competition**: BirdCLEF 2025
**Score**: N/A (pseudo-labeling utility)

## Summary

This notebook generates pseudo-labels for unlabeled soundscape recordings by applying a trained model and extracting high-confidence predictions. The output is a training-compatible dataset with both CSV labels and pre-computed mel-spectrograms, enabling semi-supervised learning by incorporating unlabeled data into the training process.

## Approach

### Architecture

The notebook loads a pre-trained RegNetY-008 model from timm with custom classification head. The architecture uses adaptive average pooling to reduce spatial dimensions before the final linear classifier. The model supports ensemble inference by loading multiple checkpoint files and averaging their predictions.

### Data Pipeline

Audio processing converts soundscape recordings to mel-spectrograms using librosa with the following configuration:
- Sample rate: 32kHz
- Window size: 5 seconds per segment
- FFT parameters: 1024 bins, 512 hop length
- Mel filterbank: 128 bands, 50-14000 Hz frequency range
- Target shape: 256x256 via OpenCV resize

The pipeline handles NaN values by replacing them with the mean, normalizes spectrograms to 0-1 range using min-max scaling, and pads short segments to the target duration.

### Inference Process

The notebook processes unlabeled soundscapes in 5-second windows:
1. Load audio file and resample to 32kHz
2. Segment into non-overlapping 5-second chunks
3. Convert each segment to mel-spectrogram
4. Run model inference (with optional ensemble averaging)
5. Apply sigmoid activation to get class probabilities
6. Extract predictions above confidence threshold (0.93)

Only segments where the maximum predicted probability exceeds the threshold become pseudo-labels. This conservative approach prioritizes precision over recall.

### Output Format

The notebook produces two outputs:
1. CSV file mimicking the competition's train.csv format with filename, primary_label, and metadata columns
2. NPY file containing a dictionary mapping segment identifiers to pre-computed mel-spectrograms

The pre-computed spectrograms enable direct loading during training without reprocessing, saving compute time in subsequent training runs.

## Key Ideas

- **High-confidence filtering**: Using a 0.93 probability threshold ensures pseudo-labels have high precision, reducing label noise in semi-supervised training
- **Spectrogram caching**: Pre-computing and saving mel-spectrograms in a dictionary structure avoids redundant audio processing during training
- **Train.csv compatibility**: Structuring pseudo-label output to match the competition's training data format allows seamless integration with existing data loaders
- **Ensemble averaging**: Supporting multiple checkpoint files and averaging their predictions produces more robust pseudo-labels than single-model inference
- **Segment-level granularity**: Processing soundscapes in 5-second windows matches typical training segment duration, creating consistent data distribution

## Results

This is a utility notebook for generating training data rather than producing competition submissions. The effectiveness depends on downstream training experiments that incorporate the pseudo-labeled data.

## Connections

- **Related concepts**: Semi-supervised learning, knowledge distillation, self-training
- **Informs experiments**: Could be adapted for Experiment 010 (teacher labels) by using ensemble models as high-confidence teachers
- **Similar approaches**: BirdNET/Perch in Experiment 010 use similar confidence thresholding for automatic labeling, though with different model architectures
