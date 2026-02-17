---
slug: "stefankahl/birdclef-2025-sample-submission"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 245
---

# Kaggle Summary: BirdCLEF+ 2025 Sample Submission

**Source**: https://www.kaggle.com/code/stefankahl/birdclef-2025-sample-submission
**Competition**: BirdCLEF+ 2025
**Score**: N/A (baseline/template notebook)

## Summary

This is the official competition baseline notebook demonstrating the minimal submission format and workflow. It shows how to process test soundscapes, generate predictions for 5-second chunks, and create a valid submission file without using any actual model.

## Approach

### Architecture

No real model is implemented. The notebook uses random number generation as a placeholder for model predictions, demonstrating that any scoring function that outputs probabilities for all 206 classes is valid.

### Data Pipeline

The pipeline walks through the test soundscape directory and loads each audio file using librosa at its native sample rate. Each soundscape is segmented into non-overlapping 5-second chunks by iterating through the audio signal in fixed-size windows based on the sample rate.

### Training

No training occurs in this notebook. It is purely a submission template.

### Inference / Post-Processing

For each 5-second chunk, the notebook generates a row ID by concatenating the soundscape filename (without extension) with the end timestamp of the chunk (e.g., "soundscape_001_5", "soundscape_001_10", etc.). Random scores are generated for all 206 class labels and appended to a growing DataFrame. The final DataFrame containing all row IDs and class predictions is saved as a CSV file.

## Key Ideas

- **Submission format**: The competition requires predictions for every 5-second window of every test soundscape, identified by `{soundscape_id}_{end_time}` row IDs
- **All 206 classes required**: Every prediction row must contain scores for all 206 species from the training set, regardless of which species are actually expected in that soundscape
- **Internet must be disabled**: The notebook settings must have internet turned off for a valid competition submission
- **90-minute time limit**: Submissions have a hard runtime limit, so efficiency matters for real models
- **Hidden test set**: Test soundscapes are only visible during submission execution, preventing manual inspection or overfitting to specific test files

## Results

| Metric | Value |
|--------|-------|
| CV     | N/A   |
| Public LB | N/A (random baseline) |
| Private LB | N/A |

## Connections

- **Related concepts**: This establishes the baseline submission format that all experimental models must follow
- **Informs experiments**: All inference pipelines (exp 009 onwards) must output predictions in this exact format with matching row IDs
- **Similar approaches**: This is the canonical baseline provided by competition organizers
