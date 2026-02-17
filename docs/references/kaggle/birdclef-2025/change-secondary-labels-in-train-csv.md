---
slug: "tomkkk/change-secondary-labels-in-train-csv"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 22
---

# Kaggle Summary: Change Secondary Labels in train.csv

**Source**: https://www.kaggle.com/code/tomkkk/change-secondary-labels-in-train-csv
**Competition**: BirdCLEF 2025
**Score**: Not reported

## Summary

This notebook presents a multi-stage workflow for augmenting the training dataset's secondary labels using model predictions on the training data itself. The approach runs inference on training audio to identify confident predictions above a threshold, then enriches the secondary labels field with these detected species to create pseudo-labels that can improve model training.

## Approach

### Architecture

The inference pipeline uses an EfficientNet-B0 backbone from the timm library. The model architecture follows a standard pattern: the backbone's classification head is replaced with global adaptive average pooling followed by a linear classifier for multi-label bird species prediction.

### Data Pipeline

Audio processing converts recordings to mel spectrograms with the following parameters:
- Sample rate: 32kHz
- Window size: 5 seconds per segment
- FFT size: 1034 (note: likely intended to be 1024)
- Hop length: 64 samples
- Mel bins: 136
- Frequency range: 20Hz to 16kHz
- Target shape: 256x256 pixels after resizing

The pipeline handles variable-length audio by extracting the center portion of each recording, padding if necessary to reach the 5-second window size. Spectrograms are normalized using min-max scaling after converting power to decibels.

### Training

The notebook focuses on the inference and label augmentation workflow rather than training details. The trained model checkpoint is loaded from a pre-existing source. The pipeline supports both single-model and ensemble prediction modes.

### Inference / Post-Processing

The inference workflow includes several configurable options:
- Test-time augmentation (TTA) with horizontal and vertical flips of spectrograms
- Multi-model ensemble averaging when multiple checkpoints are available
- Temporal smoothing using weighted averaging of neighboring segments (weights: 0.2 for neighbors, 0.6-0.8 for current segment)
- Probability threshold filtering (default 0.7) to select confident predictions

The label augmentation process:
1. Runs inference on training audio files to generate per-species probability predictions
2. Filters predictions above a confidence threshold (e.g., 0.7)
3. Excludes the primary label from secondary label candidates
4. Either replaces or appends to existing secondary labels based on configuration
5. Writes the modified training CSV for use in subsequent training runs

## Key Ideas

- **Self-supervised label enrichment**: Using model predictions on training data to augment secondary labels creates additional supervision signal without manual annotation effort
- **Confidence-based filtering**: Only predictions above a probability threshold are used to augment labels, reducing noise from uncertain predictions
- **Flexible augmentation modes**: The workflow supports both replacing existing secondary labels entirely or appending new discoveries to existing labels
- **Multi-stage pipeline**: The approach separates inference (Python) and label manipulation (R script) into distinct stages, allowing for iterative refinement
- **Temporal consistency**: The smoothing operation enforces that predictions for adjacent time segments should be similar, which aligns with the expectation that bird vocalizations persist across multiple 5-second windows

## Results

Not reported in the notebook. The notebook serves as a data augmentation utility rather than a complete training pipeline.

## Connections

- **Related concepts**: This approach relates to pseudo-labeling and knowledge distillation techniques where model predictions are used to enhance training data
- **Informs experiments**: The secondary label enrichment strategy could be combined with teacher model approaches (see experiment 010) to create richer training supervision
- **Similar approaches**: The temporal smoothing technique is a common post-processing strategy in soundscape competitions to improve prediction consistency
