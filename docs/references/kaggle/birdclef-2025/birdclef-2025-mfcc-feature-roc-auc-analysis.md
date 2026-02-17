---
slug: "jocelyndumlao/birdclef-2025-mfcc-feature-roc-auc-analysis"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 146
---

# Kaggle Summary: BirdCLEF 2025 MFCC Feature & ROC-AUC Analysis

**Source**: https://www.kaggle.com/code/jocelyndumlao/birdclef-2025-mfcc-feature-roc-auc-analysis
**Competition**: BirdCLEF 2025
**Score**: Unknown

## Summary

This notebook presents a baseline exploratory data analysis and simple classification approach for the BirdCLEF 2025 competition using traditional audio features. The author focuses on MFCC-based feature extraction combined with gradient boosting models, evaluating performance using macro-averaged ROC-AUC. The notebook emphasizes visualization and understanding the competition data structure rather than achieving competitive leaderboard performance.

## Approach

### Architecture

The notebook implements a traditional machine learning approach rather than deep learning. The author provides two model options: XGBoost and LightGBM classifiers. Both models use the same feature extraction pipeline but differ in their gradient boosting implementations. The XGBoost classifier is configured with disabled label encoding and log loss evaluation metric. LightGBM is used with default parameters plus a fixed random seed for reproducibility.

### Data Pipeline

Audio processing relies on librosa for loading and feature extraction. The pipeline extracts MFCC features from audio files by computing the mel-frequency cepstral coefficients and averaging them across the time dimension to produce fixed-length feature vectors. The default configuration uses 20 MFCC coefficients with a 22050 Hz sampling rate. Training data is randomly sampled (500 examples by default) to create a manageable subset for experimentation. Files are preprocessed to extract secondary labels using regular expressions and construct full file paths by joining directory paths with filenames.

### Training

The training process uses an 80/20 train-test split with a fixed random seed. For computational efficiency, only a subset of the training data is used. Primary labels serve as the classification targets. The notebook includes comprehensive data exploration visualizations: waveforms and spectrograms for individual recordings, soundscape visualizations, geographic distribution maps using Folium, and taxonomy distribution plots showing class frequency. Error handling is implemented throughout the feature extraction pipeline to gracefully skip corrupted or missing audio files.

### Inference / Post-Processing

For submission generation, the notebook iterates through the sample submission file structure and extracts MFCC features from test soundscapes. The trained model predicts class probabilities for each bird species. When audio files cannot be processed, a default small probability (0.01) is assigned to all classes. The submission format matches the required structure with row IDs and per-class probability columns. The notebook includes error handling for missing files and key errors during prediction.

## Key Ideas

- Simple MFCC features can establish a baseline for audio classification without deep learning infrastructure
- Macro-averaged ROC-AUC evaluation properly handles multi-class imbalance by computing per-class AUC and averaging
- Data exploration through waveform, spectrogram, and geographic visualizations helps understand the competition dataset
- MultiLabelBinarizer enables proper one-vs-all evaluation even though primary labels are single-class
- Default probability assignment (0.01) for failed predictions ensures complete submission files
- The notebook trains on a small subset (500 samples) making it accessible for quick experimentation without GPU resources
- Geographic density mapping visualizes the spatial distribution of bird observations with interpolated radius scaling
- Model type selection between XGBoost and LightGBM allows easy comparison of gradient boosting implementations

## Results

| Metric | Value |
|--------|-------|
| CV     | Not reported |
| Public LB | Not reported |
| Private LB | Not reported |

The notebook computes macro-averaged ROC-AUC on the validation split and plots per-class ROC curves, but specific numerical scores are not documented.

## Connections

- **Related concepts**: Traditional audio features (MFCCs) as baseline compared to spectrogram-based deep learning approaches
- **Informs experiments**: Provides baseline evaluation methodology and data exploration patterns
- **Similar approaches**: Contrasts with deep learning approaches using CNNs or transformers on mel-spectrograms
