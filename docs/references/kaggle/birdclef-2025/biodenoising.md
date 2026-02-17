---
slug: "samvelkoch/biodenoising"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 26
---

# Kaggle Summary: Biodenoising

**Source**: https://www.kaggle.com/code/samvelkoch/biodenoising
**Competition**: BirdCLEF 2025
**Score**: N/A (demonstration notebook, not a submission)

## Summary

This notebook demonstrates applying a pre-trained audio denoising model specifically designed for animal vocalizations to BirdCLEF training data. The approach leverages transfer learning from speech enhancement models rather than requiring clean animal audio for training, making it practical for bioacoustic applications where clean recordings are scarce.

## Approach

### Architecture

The notebook uses the Biodenoising model, a neural denoising system built on principles from speech enhancement research. The specific variant used is `biodenoising16k_dns48`, which operates at 16 kHz sample rate and was trained using the DNS (Deep Noise Suppression) framework. The model does not require clean animal audio during training, instead transferring knowledge from speech denoising tasks.

### Data Pipeline

Audio files are loaded from the BirdCLEF 2025 training set and converted to the model's expected format (16 kHz, mono). The notebook demonstrates processing on both individual examples and a random sample of 10 training files. Mel-spectrograms are computed before and after denoising for visual comparison using 1024-point FFT, 512-sample hop length, and 128 mel bins.

### Training

No training is performed in this notebook. It uses a fully pre-trained model downloaded from the biodenoising package. The pre-training methodology (described in the referenced paper) involves self-supervised learning on noisy animal recordings without requiring paired clean/noisy examples.

### Inference / Post-Processing

Inference is straightforward: load audio, resample to 16 kHz, pass through the model, and obtain denoised output. The notebook focuses on exploration and visualization rather than submission generation. Mel-spectrogram visualizations allow qualitative assessment of noise reduction effectiveness across different recordings.

## Key Ideas

- **Transfer learning from speech to bioacoustics**: Speech enhancement models contain useful priors about temporal structure in audio that generalize to animal vocalizations, even though the acoustic characteristics differ significantly
- **Unsupervised denoising**: The method does not require clean reference audio for training, addressing a major bottleneck in bioacoustic research where pristine recordings are difficult to obtain
- **Cross-species generalization**: A single model can denoise diverse animal species by learning signal patterns that recur across different temporal scales and frequency ranges, rather than requiring species-specific training
- **Preprocessing exploration**: While not a complete submission, the notebook suggests denoising as a potential preprocessing step to improve model training by reducing environmental noise in training data

## Results

No quantitative metrics are reported. The notebook is exploratory, demonstrating the denoising model's capabilities through audio playback and mel-spectrogram visualizations on BirdCLEF data.

| Metric | Value |
|--------|-------|
| CV     | N/A   |
| Public LB | N/A |
| Private LB | N/A |

## Connections

- **Related concepts**: Could inform strategies for data quality improvement, particularly for soundscape recordings with high background noise
- **Potential applications**:
  - Preprocessing step before feature extraction or training
  - Data augmentation by creating denoised variants
  - Analysis of noise characteristics in different recording environments
- **External reference**:
  - GitHub: https://github.com/earthspecies/biodenoising/
  - Paper: https://arxiv.org/abs/2410.03427
  - Earth Species Project initiative for bioacoustic signal processing
