---
slug: "johnyim1570/bird25-weightedblend-nfnet-seresnext-0-878"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 241
---

# Kaggle Summary: Bird25 WeightedBlend NFNet + SEResNeXt

**Source**: https://www.kaggle.com/code/johnyim1570/bird25-weightedblend-nfnet-seresnext-0-878
**Competition**: BirdCLEF 2025
**Score**: 0.878 (Public LB)

## Summary

This notebook achieves a strong public leaderboard score of 0.878 through a weighted ensemble of two different sound event detection (SED) models: an ECA-NFNet-L0 model and a SEResNeXt26t model. The approach combines different architectural families with asymmetric weighting, post-processing techniques to boost low-confidence predictions, and temporal smoothing across adjacent 5-second segments.

## Approach

### Architecture

The notebook uses two distinct SED architectures:

1. **NFNet Model (Submission 1)**: Based on the ECA-NFNet-L0 backbone, a variant of NFNets (Normalizer-Free Networks) with Efficient Channel Attention. The model uses a 3-fold ensemble of identical architectures trained on different folds.

2. **SEResNeXt Model (Submission 2)**: Built on the SEResNeXt26t_32x4d backbone, which combines ResNeXt group convolutions with Squeeze-and-Excitation blocks for channel attention. This model uses a single checkpoint.

Both models share a common SED head design:
- Mel-spectrogram features are processed through a CNN backbone
- Features are averaged across the frequency dimension to create time-series embeddings
- Channel smoothing via max-pooling and average-pooling with kernel size 3
- A fully-connected layer for feature transformation
- An attention-based pooling mechanism (AttBlockV2) that learns both attention weights and class-specific transformations
- Final predictions combine weighted attention over time steps

### Data Pipeline

**NFNet Pipeline**:
- Audio loaded via torchaudio at 32kHz sampling rate
- Each 60-second soundscape is divided into twelve 5-second segments
- Mel-spectrograms computed with: n_fft=1024, hop_length=512, n_mels=128, frequency range 50-16000 Hz
- Log-scaling applied with small epsilon for numerical stability
- Per-spectrogram normalization using mean and standard deviation
- Single-channel spectrograms replicated to 3 channels for compatibility with ImageNet-pretrained backbones

**SEResNeXt Pipeline**:
- Similar 32kHz sampling rate and 5-second target segments
- Training uses 10-second duration audio with padding/cropping during inference to create context
- For each 5-second target, a 10-second window is extracted with the target centered, providing 2.5 seconds of context on each side
- Edge segments have padding zeroed out to avoid boundary artifacts
- Mel-spectrograms configured with similar parameters to NFNet pipeline
- Amplitude-to-decibel conversion with 80 dB dynamic range
- Normalization by dividing by 80 after adding 80 dB offset

### Training

The notebook uses pre-trained model weights rather than training from scratch:

**NFNet weights**: 3 model checkpoints from a prior training run, suggesting 3-fold cross-validation
**SEResNeXt weights**: Single checkpoint from a separate training run

Training configurations are embedded in the checkpoints but include:
- Loss function: Binary cross-entropy with attention-pooled predictions
- Dropout rate: 0.2 (both feature dropout and drop-path)
- Models trained with mixed precision (half/float operations during inference)

### Inference / Post-Processing

The inference pipeline applies several post-processing techniques:

1. **Power Transformation for Low-Confidence Species**: A function (`apply_power_to_low_ranked_cols`) identifies species whose maximum prediction across all time steps ranks below the top 30. For these "tail" species, predictions are raised to the power of 2, effectively boosting weak signals while suppressing noise. This addresses the long-tail distribution of bird species in the wild.

2. **Test-Time Augmentation (SEResNeXt only)**: The SEResNeXt model uses temporal TTA by extracting features at three different time positions (original, shifted earlier by 2 time steps, shifted later by 2 time steps) and averaging predictions with weights 0.5, 0.25, 0.25.

3. **Temporal Smoothing**: After generating predictions for each 5-second segment, a smoothing operation averages predictions with adjacent segments:
   - First segment: 80% current + 20% next
   - Middle segments: 20% previous + 60% current + 20% next
   - Last segment: 80% current + 20% previous

   This enforces temporal consistency based on the assumption that bird presence changes gradually.

4. **Weighted Ensemble**: Final predictions combine the two model outputs:
   - SEResNeXt: 75% weight
   - NFNet: 25% weight

   This asymmetric weighting favors the SEResNeXt model, which likely had better validation performance.

5. **Parallel Processing**: Both models use ThreadPoolExecutor with 4-5 workers to parallelize inference across soundscape files.

### Model Ensemble Strategy

The two-stage ensemble approach first averages predictions within each model family (3 NFNet folds are averaged), then blends across architectures. This hierarchical ensemble strategy allows different architectural inductive biases to complement each other while reducing variance from individual training runs.

## Key Ideas

- **Asymmetric ensemble weighting**: Rather than equal weighting, the 75/25 split suggests empirical tuning to favor the stronger model while still benefiting from diversity.

- **Power transformation for long-tail species**: The top_k=30 threshold and exponent=2 are clever hyperparameters that address the class imbalance problem without requiring species-specific tuning. By identifying low-confidence species dynamically, the method adapts to each soundscape.

- **Contextual audio windows**: The SEResNeXt model's 10-second window for 5-second predictions provides temporal context that may help distinguish similar vocalizations and reduce false positives from transient noise.

- **Attention-based temporal aggregation**: The AttBlockV2 module learns both what features to attend to (via attention weights) and how to transform them (via class-specific convolutions), providing more flexibility than simple max/mean pooling.

- **Multi-scale temporal smoothing**: The smoothing weights (0.2/0.6/0.2 for interior segments) create a triangular kernel that balances local prediction accuracy with temporal consistency.

## Results

| Metric | Value |
|--------|-------|
| CV     | Not reported |
| Public LB | 0.878 |
| Private LB | Not reported |

**Note**: The author acknowledges potential overfitting to the public test set, noting that high-scoring ensembles in BirdCLEF 2022-2024 often experienced significant shake-ups between public and private leaderboards. The 0.878 score should be interpreted cautiously as it may not generalize to the final evaluation.

## Connections

- **Related concepts**: Ensemble learning, sound event detection, attention mechanisms, long-tail classification
- **Similar approaches**: This builds on earlier BirdCLEF ensemble techniques and references two public notebooks:
  - Single SED model inference (0.857 LB baseline)
  - Post-processing with power adjustment for low-rank predictions
  - NFNet + ConvNeXtV2 weighted blend (0.860 LB)
- **Architectural diversity**: Combines NFNets (normalizer-free) with SEResNeXt (residual connections + group convolutions + squeeze-excitation), providing complementary feature representations
