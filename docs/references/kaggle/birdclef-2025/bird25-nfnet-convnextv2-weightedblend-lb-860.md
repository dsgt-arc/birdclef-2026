---
slug: "hideyukizushi/bird25-nfnet-convnextv2-weightedblend-lb-860"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 436
---

# Kaggle Summary: Bird25 NFNet+ConvNeXtV2 Weighted Blend

**Source**: https://www.kaggle.com/code/hideyukizushi/bird25-nfnet-convnextv2-weightedblend-lb-860
**Competition**: BirdCLEF 2025
**Score**: 0.860 Public LB

## Summary

This notebook achieves strong performance through model diversity, combining predictions from two different backbone architectures using weighted blending. The approach demonstrates that ensemble methods using architecturally distinct models can improve robustness compared to single-architecture solutions. The inference pipeline includes temporal smoothing post-processing and a power adjustment technique applied to low-ranked predictions.

## Approach

### Architecture

**Model 1 (ConvNeXtV2-Nano)**:
- Backbone: ConvNeXtV2-Nano from timm library
- Pre-training: FCMAE-pretrained on ImageNet-22k, fine-tuned on ImageNet-1k
- Head: AdaptiveAvgPool2d followed by linear classifier
- Parameters: Single-channel input converted from mel spectrogram

**Model 2 (ECA-NFNet-L0)**:
- Backbone: ECA-NFNet-L0 from timm library
- Architecture: Sound Event Detection (SED) model with attention pooling
- Head: AttBlockV2 module using dual Conv1d branches for attention weights and class activations
- Temporal aggregation: Combination of max pooling and average pooling along time dimension
- FC layer with ReLU activation before attention block

### Data Pipeline

**Audio Processing**:
- Sample rate: 32 kHz
- Window size: 5-second non-overlapping segments
- ConvNeXtV2 pipeline:
  - Mel spectrogram: 512 mel bins, FFT size 2048, hop length 512
  - Frequency range: 20-16000 Hz
  - Power-to-dB conversion with min-max normalization
  - Resized to 256x256 using bilinear interpolation
- NFNet pipeline:
  - Mel spectrogram: 128 mel bins, FFT size 1024, hop length 512
  - Frequency range: 50-16000 Hz
  - Log-scaled with per-spectrogram standardization (mean/std normalization)
  - 1-channel input replicated to 3 channels
- Both use librosa/torchaudio with HTK mel scale and Slaney normalization

**Test-time augmentation** (optional, disabled in submitted version):
- Horizontal flip (time reversal)
- Vertical flip (frequency reversal)

### Training

Details not included in this inference-only notebook. Models were trained externally with:
- ConvNeXtV2: Custom training (author's local model)
- NFNet: SED-style training with attention-based temporal pooling

Ensemble approach:
- ConvNeXtV2: Multiple folds (folds 0 and 1) loaded and averaged
- NFNet: Three separate model checkpoints averaged

### Inference / Post-Processing

**Ensemble Strategy**:
- Two-stage blending:
  1. Within each architecture: Average predictions across multiple model checkpoints/folds
  2. Cross-architecture: Weighted blend of 0.6 NFNet + 0.4 ConvNeXtV2

**Power Adjustment for Low-Ranked Predictions**:
- Applied only to NFNet predictions before blending
- Ranks all species by their maximum prediction across time chunks
- Top 30 species remain unchanged
- All species outside top 30 are raised to power of 2 (squared)
- Rationale: Suppresses low-confidence predictions for less-likely species while preserving high-confidence detections

**Temporal Smoothing**:
- Applied to both models independently before final blending
- Weighted moving average across consecutive 5-second segments:
  - Middle chunks: 0.2 × previous + 0.6 × current + 0.2 × next
  - First chunk: 0.9 × current + 0.1 × next
  - Last chunk: 0.9 × current + 0.1 × previous
- Smooths temporal predictions to reduce flickering between adjacent segments

**Inference Pipeline**:
- CPU-only execution for Kaggle submission constraints
- Concurrent audio loading for NFNet pipeline (ThreadPoolExecutor with 5 workers)
- Per-segment processing with sigmoid activation on logits
- Final submission includes all 182 species with proper column alignment

## Key Ideas

- **Model diversity over single-architecture ensembles**: Combining ConvNeXtV2 (hierarchical vision transformer) and NFNet (attention-based SED) provides complementary prediction patterns
- **Asymmetric weighting favors NFNet**: 60/40 split suggests the SED-style architecture with temporal attention performed better than the image classification approach
- **Power adjustment targets long-tail suppression**: Squaring low-ranked predictions reduces false positives for rare species while maintaining sensitivity for top detections
- **Temporal smoothing improves consistency**: Weighted averaging across adjacent chunks assumes bird calls persist across multiple 5-second windows
- **Different mel spectrogram parameters per model**: ConvNeXtV2 uses higher resolution (512 mels) while NFNet uses standard audio parameters (128 mels), suggesting models were trained with different input representations

## Results

| Metric | Value |
|--------|-------|
| CV     | Not reported |
| Public LB | 0.860 |
| Private LB | Not reported |

## Connections

- **Related concepts**:
  - Ensemble methods and model diversity
  - Sound event detection architectures
  - Temporal post-processing for sequential predictions
- **Informs experiments**:
  - Potential application of power adjustment to our SSAMBA predictions
  - Temporal smoothing as simple post-processing improvement
  - Multi-architecture ensembling beyond single-model families
- **Similar approaches**:
  - NFNet component references another public notebook (myso1987/post-processing-with-power-adjustment-for-low-rank)
  - Author mentions previous 0.829 LB notebook using focal loss + BCE
