---
slug: "carlolepelaars/generating-audio-embeddings-with-auditus"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 35
---

# Kaggle Summary: Generating Audio Embeddings with Auditus

**Source**: https://www.kaggle.com/code/carlolepelaars/generating-audio-embeddings-with-auditus
**Competition**: BirdCLEF 2025
**Score**: N/A (exploratory notebook)

## Summary

This notebook demonstrates how to generate fixed-length audio embeddings for the entire BirdCLEF 2025 training dataset using Google's Bird Vocalization Classifier model and the auditus library. The approach produces 1280-dimensional embedding vectors that can be used for downstream analysis, visualization, or as features for classification models.

## Approach

### Architecture

The notebook uses Google's pre-trained Bird Vocalization Classifier model as a frozen feature extractor. This is a TensorFlow model that produces 1280-dimensional embeddings from audio inputs. The model was designed for bird vocalization analysis and is accessed through the auditus library, which provides a simple pipeline interface for audio processing.

### Data Pipeline

The audio processing pipeline consists of three main stages:

1. **Loading**: Audio files are loaded at their native sample rate of 32 kHz
2. **Truncation**: Each audio file is truncated or padded to exactly 5 seconds (160,000 samples at 32 kHz), since the Bird Vocalization Classifier requires fixed-length input. Padding with zeros is applied if the audio is shorter than 5 seconds.
3. **Embedding extraction**: The preprocessed audio is passed through the TensorFlow model to extract a single 1280-dimensional embedding vector

The pipeline processes both short training clips and longer soundscape recordings. For simplicity, only the first 5 seconds of each file are used, discarding any information beyond that point.

### Training

No training is performed - this is purely feature extraction using a pre-trained model. The notebook processes the full training dataset using GPU acceleration with thread pool parallelization for CPU-bound operations. The complete processing takes approximately 10 minutes for the entire training set.

### Inference / Post-Processing

The extracted embeddings are appended as a new column to the training metadata CSV file. Each row contains a list representation of the 1280-dimensional vector. The resulting dataset is saved for downstream use in visualization or modeling tasks.

## Key Ideas

- **Leveraging pre-trained bird-specific models**: Using a model specifically trained for bird vocalizations provides more relevant features than general-purpose audio models
- **Fixed-length simplification**: Truncating all audio to 5 seconds avoids the complexity of variable-length sequence handling, though it discards potentially useful information from longer recordings
- **Embeddings as features**: Rather than training an end-to-end model, this approach treats embeddings as input features that can be used for various downstream tasks including visualization, clustering, or classification
- **Pipeline abstraction**: The auditus library provides a clean functional pipeline interface that chains audio transformations and model inference
- **Exploratory foundation**: The notebook serves as a starting point for data exploration rather than a competitive submission, with embeddings enabling UMAP visualization and pattern discovery

## Results

| Metric | Value |
|--------|-------|
| Embedding dimensionality | 1280 |
| Processing time (full dataset) | ~10 minutes |
| Input length | 5 seconds (fixed) |
| Sample rate | 32 kHz |

## Connections

- **Related concepts**: This approach is relevant to transfer learning, feature extraction, and audio representation learning
- **Informs experiments**: Could inform experiment designs using pre-trained embeddings as input features rather than raw spectrograms
- **Similar approaches**: Related to other embedding-based methods like PERCH and BirdNET that produce fixed-size representations from audio
- **Dataset output**: Creates the "birdclef-2025-perch-embeddings" Kaggle dataset for downstream visualization and analysis
