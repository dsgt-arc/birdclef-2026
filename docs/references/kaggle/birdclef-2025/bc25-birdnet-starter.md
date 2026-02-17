---
slug: "kdmitrie/bc25-birdnet-starter"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 34
---

# Kaggle Summary: BC25 BirdNet Starter

**Source**: https://www.kaggle.com/code/kdmitrie/bc25-birdnet-starter
**Competition**: BirdCLEF 2025
**Score**: Baseline submission (no reported score)

## Summary

A minimalist starter notebook demonstrating how to use the pre-trained BirdNET model for zero-shot bird species detection in soundscape recordings. The approach requires no training and produces predictions by running BirdNET's analyzer on test audio, then temporally resampling the outputs to match the competition's submission format.

## Approach

### Architecture

The notebook uses BirdNET, a pre-trained audio classification model accessed through the `birdnetlib` Python wrapper. BirdNET was originally trained on a large corpus of bird vocalizations and provides out-of-the-box predictions for hundreds of bird species. No model training or fine-tuning is performed - this is purely inference with the existing checkpoint.

### Data Pipeline

Audio files are processed directly from the competition's test soundscape directory (or train soundscapes if test is unavailable, for local validation). The BirdNET analyzer processes entire 60-second soundscape files and returns detections with timestamps and confidence scores. A minimum confidence threshold of 1e-10 is used to capture all possible detections.

The notebook includes a mapping from BirdNET's common bird names to the competition's species taxonomy using the provided taxonomy CSV file.

### Training

No training is performed. This is a zero-shot inference approach using BirdNET's pre-trained weights.

### Inference / Post-Processing

BirdNET's analyzer produces detections with arbitrary time boundaries. The notebook performs temporal resampling to align predictions with the competition format:

1. BirdNET outputs are initially placed into 3-second bins (matching BirdNET's native temporal resolution)
2. These are then resampled to 5-second intervals using cubic spline interpolation to match the competition's required output format
3. A zero-filled array is created for all target species, with BirdNET confidences filled in for detected species
4. The final submission assigns each 5-second segment a row ID following the pattern `{soundscape_id}_{end_time}`

A timeout mechanism is implemented (5300 seconds) to ensure the kernel completes within Kaggle's time limits.

## Key Ideas

- Demonstrates that BirdNET provides a strong zero-shot baseline without any custom training
- Cubic spline interpolation elegantly handles the temporal resolution mismatch between BirdNET's native 3-second windows and the competition's 5-second requirement
- The taxonomy mapping between BirdNET's common names and competition species codes is a critical preprocessing step
- Suppressing BirdNET's verbose output using a context manager keeps the notebook output clean
- Setting an extremely low confidence threshold (1e-10) ensures all possible detections are captured, leaving the scoring system to handle weak predictions

## Results

| Metric | Value |
|--------|-------|
| CV     | Not reported |
| Public LB | Not reported |
| Private LB | Not reported |

This is a starter baseline notebook rather than a competitive solution. Its primary value is demonstrating the BirdNET integration and providing a working submission template.

## Connections

- **Related concepts**: Transfer learning, zero-shot classification, temporal resampling
- **Informs experiments**: Could serve as a weak teacher for pseudo-labeling or as an ensemble component
- **Similar approaches**: Other BirdNET-based submissions in BirdCLEF competitions
