---
slug: "myso1987/birdclef2025-2-train-baseline-5s"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 132
---

# Kaggle Summary: BirdCLEF2025 Train Baseline 5s

**Source**: https://www.kaggle.com/code/myso1987/birdclef2025-2-train-baseline-5s
**Competition**: BirdCLEF 2025
**Score**: Unknown (training notebook only)

## Summary

A straightforward ResNet34-based training pipeline using 5-second fixed-length audio segments. The approach demonstrates a simple but effective baseline for the BirdCLEF 2025 competition, using standard computer vision architectures adapted for mel spectrogram classification.

## Approach

### Architecture

The model uses a pretrained ResNet34 backbone from torchvision. The single-channel mel spectrogram is replicated three times to match the expected RGB input format. The original ImageNet classification head is replaced with a linear layer outputting predictions for all bird species classes in the competition.

### Data Pipeline

Audio is preprocessed into 5-second segments at 32 kHz sampling rate. Mel spectrograms are computed using 1024-sample FFT with 512-sample hop length, covering a frequency range of 20-15000 Hz and producing 128 mel bins. The pipeline applies per-spectrogram standardization by subtracting the mean and dividing by the standard deviation. A small amount of random noise is added to the waveform before spectrogram computation as a simple augmentation technique. The dataset is split 80/20 for training and validation using a fixed random seed.

### Training

The model is trained for 8 epochs using cross-entropy loss with Adam optimizer at a learning rate of 0.001. Batch size is set to 24 samples. The training loop applies softmax to model outputs and evaluates using the competition's custom ROC-AUC metric. The model starts from ImageNet pretrained weights to provide a better initialization than training from scratch.

### Inference / Post-Processing

This is a training-only notebook. A separate inference notebook is referenced that handles the actual submission generation.

## Key Ideas

- **Fixed 5-second segments**: Preprocessing splits longer recordings into uniform 5-second chunks, simplifying the training pipeline and allowing consistent batch processing
- **Grayscale-to-RGB replication**: Simple technique to adapt single-channel spectrograms for pretrained vision models without modifying the architecture
- **Per-sample normalization**: Standardizing each spectrogram individually helps handle variations in recording volume and background noise levels
- **Direct softmax classification**: Treats the problem as single-label classification (one species per clip) rather than multi-label, matching the structure of preprocessed 5-second segments

## Results

| Metric | Value |
|--------|-------|
| CV     | Not reported in notebook |
| Public LB | Not applicable (training only) |
| Private LB | Not applicable (training only) |

**Training metrics** (final epoch):
- Training AUC: High (notebook shows training progression)
- Validation AUC: Reported per epoch during training

## Connections

- **Related concepts**: Simple baseline approach, good starting point for understanding competition data structure
- **Informs experiments**: Demonstrates standard mel spectrogram pipeline and ResNet adaptation pattern
- **Similar approaches**: Most BirdCLEF baselines follow similar patterns with pretrained vision models
