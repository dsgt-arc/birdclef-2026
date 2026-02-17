---
slug: "myso1987/birdclef2025-sound-event-visualisations"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 530
---

# Kaggle Summary: BirdCLEF2025+: Sound Event Visualisations

**Source**: https://www.kaggle.com/code/myso1987/birdclef2025-sound-event-visualisations
**Competition**: BirdCLEF 2025
**Score**: N/A (visualization tool, not a submission)

## Summary

This notebook provides a visualization tool for understanding sound event detection model predictions on BirdCLEF 2025 soundscape data. It displays mel spectrograms alongside temporal heatmaps of species predictions, allowing qualitative inspection of when and where models detect bird calls in 60-second soundscape recordings.

## Approach

### Architecture

This is not a training or inference notebook itself, but rather uses an external utility (`birdclef2025_utils.get_results`) to generate predictions from a pre-trained model. The notebook references a companion training notebook for a 5-second baseline model.

### Data Pipeline

Audio processing for visualization uses torchaudio transforms with the following parameters:
- Sample rate: 32 kHz
- FFT parameters: n_fft=1024, win_length=1024, hop_length=512
- Mel spectrogram: 128 mel bins, frequency range 20-16000 Hz
- Normalization: Slaney norm with HTK mel scale
- Power-to-dB conversion with 10*log10 scaling

The notebook operates in two modes:
1. **Test mode**: Processes actual test soundscapes from `/test_soundscapes/`
2. **Debug mode**: When test soundscapes are unavailable (e.g., pre-competition), uses a subset of train soundscapes for visualization (configurable start index and count)

### Training

N/A - This is a visualization-only notebook.

### Inference / Post-Processing

The notebook generates predictions for 60-second soundscape files and formats results as:
- Row format: one row per 5-second window (12 windows per 60-second file)
- Column format: one column per species class
- Values: Predicted probabilities in [0, 1] range

## Key Ideas

- **Dual visualization approach**: Combines mel spectrogram (acoustic features) with temporal heatmap (model predictions) to enable visual correlation between acoustic events and detections
- **Multi-model comparison**: The notebook is designed to load and compare predictions from multiple submission CSV files at different leaderboard scores (0.815, 0.854, 0.883, 0.903), enabling side-by-side quality assessment
- **Debug mode for pre-competition development**: Clever workaround to enable visualization during development by using train soundscapes when test soundscapes aren't available
- **Runtime estimation**: Includes timing logic to extrapolate full submission runtime from debug runs on small subsets
- **Heatmap design**: Species predictions shown as 2D heatmap with time on x-axis (12 columns for 5-second windows) and species on y-axis, with color intensity representing probability (0-1 scale, Blues colormap)

## Results

This is a visualization tool, not a scored submission. The notebook displays comparative results from multiple submissions:

| Version | Public LB |
|---------|-----------|
| Latest  | N/A       |
| Model 1 | 0.903     |
| Model 2 | 0.883     |
| Model 3 | 0.854     |
| Model 4 | 0.815     |

## Connections

- **Related concepts**: Sound event detection (SED), temporal predictions, model interpretability
- **Informs experiments**: This visualization approach could be adapted for analyzing our SSAMBA models' temporal prediction quality on soundscapes
- **Similar approaches**: This notebook is part of a series including dataset creation (crop audio 5s), training (baseline 5s), and inference (submit baseline 5s) notebooks
