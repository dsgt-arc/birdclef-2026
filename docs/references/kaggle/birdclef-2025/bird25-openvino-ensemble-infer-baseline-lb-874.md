---
slug: "kurisew/bird25-openvino-ensemble-infer-baseline-lb-874"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 375
---

# Kaggle Summary: Bird25 OpenVINO Ensemble Inference Baseline

**Source**: https://www.kaggle.com/code/kurisew/bird25-openvino-ensemble-infer-baseline-lb-874
**Competition**: BirdCLEF 2025
**Score**: 0.874 Public LB (stable performance in ~1 hour runtime)

## Summary

This notebook demonstrates a three-stage ensemble inference pipeline combining single SED (Sound Event Detection) models, OpenVINO-optimized models, and 3-fold SED models with careful weight balancing. The author discovered an optimal ordering and weighting scheme that achieves stable 0.874 LB performance while acknowledging that overfitting to public LB could reach 0.88+ but hurt private scores.

## Approach

### Architecture

The ensemble combines three distinct model groups:

1. **Single SED Model**: A custom attention-pooled SED architecture using timm backbones (seresnext26t_32x4d) with AttBlockV2 attention mechanism. Features batch normalization on mel inputs, encoder backbone, FC layer, and attention-based pooling producing both clip-wise and segment-wise outputs.

2. **OpenVINO Models**: Three CPU-optimized models (EfficientNet-B0, RegNetY-008, and EfficientNet-B4 fold-3) compiled for fast inference using OpenVINO runtime. These models operate on 256x256 mel spectrograms and support optional TTA.

3. **3-Fold SED Models**: Three ECA-NFNet-L0 models trained on different data folds, using timm architecture with AttBlockV2 attention pooling.

All models output per-species probabilities for 5-second audio segments.

### Data Pipeline

**Audio Processing**:
- Sample rate: 32kHz
- Window size: 5 seconds
- Mel spectrogram parameters vary by model group:
  - Single SED: n_fft=1024, hop_length=256, n_mels=128, fmin=48Hz, fmax=15kHz
  - OpenVINO: n_fft=1024, hop_length=256, n_mels=128, fmin=48Hz, fmax=15kHz, target shape 256x256
  - 3-fold SED: n_fft=1024, hop_length=512, n_mels=128, fmin=40Hz, fmax=16kHz

**Preprocessing Steps**:
- Librosa for audio loading and mel computation
- Pre-emphasis filtering applied
- Power-to-dB conversion with normalization
- Padding/truncation to ensure consistent 5-second chunks
- Segmentation into 12 non-overlapping 5-second chunks per minute

**TTA (Test-Time Augmentation)** - Optional for OpenVINO models:
- Original spectrogram
- Temporal shift (1/10 of time dimension)
- Gaussian noise addition
- Time masking (SpecAugment style)
- Frequency masking
- Horizontal flip with contrast enhancement

### Training

Training details are not included in this inference-only notebook, but model characteristics suggest:
- Binary cross-entropy loss (sigmoid activation on outputs)
- Attention-based pooling for temporal aggregation
- Dropout (p=0.5) for regularization
- Models accept variable-length audio through attention mechanisms

### Inference / Post-Processing

**Ensemble Strategy**:
- Three-stage sequential pipeline with weighted blending
- Ensemble weights: [0.50 (single SED), 0.35 (OpenVINO), 0.15 (3-fold SED)]
- OpenVINO models internally weighted: [0.45, 0.45, 0.10]
- 3-fold models averaged equally

**Inference Pipeline**:
1. Single SED models run first with custom infer() method using temporal attention windowing and TTA (center + shifted predictions weighted 0.5/0.25/0.25)
2. OpenVINO models run in parallel using ThreadPoolExecutor (4 workers) for multi-threaded CPU inference
3. 3-fold SED models process torchaudio mel specs with power adjustment post-processing

**Post-Processing**:
- Power adjustment applied to low-ranked species (columns not in top-30 by max score): predictions raised to power of 2 to suppress weak detections
- Optional temporal smoothing (commented out in final version): neighboring segments blended with weights [0.1, 0.8, 0.1] for middle segments, [0.9, 0.1] for edges

**Performance Optimization**:
- Concurrent file processing with ThreadPoolExecutor
- OpenVINO compilation for CPU efficiency
- Multi-threaded model inference within each file

## Key Ideas

- **Sequential ensemble ordering matters**: The author experimented with different ensemble orderings and found Single SED → OpenVINO → 3-fold SED produces the best results, differing from other popular ensembles.

- **Deliberate LB conservatism**: Author explicitly chose baseline parameters (0.874) over potentially higher-scoring overfitted configurations (0.88+) to avoid shakeup on private leaderboard, citing lessons from previous competitions.

- **Power adjustment for tail species**: Applying exponent transformations (squared predictions) to species with low maximum scores helps reduce false positives on rare/difficult classes without hurting common species.

- **Multi-stage inference with different backends**: Combining PyTorch SED models with OpenVINO-optimized models allows flexibility in compute resources while maintaining ensemble diversity.

- **TTA windowing for temporal models**: The single SED model uses temporal attention with center + delta-shifted windows (±2 time steps) to capture temporal context variations, weighted 50/25/25.

- **Temporal smoothing disabled in final**: Despite implementing neighbor-weighted smoothing, it was commented out, suggesting it may hurt or not significantly help final score.

## Results

| Metric | Value |
|--------|-------|
| CV     | Not reported |
| Public LB | 0.874 (stable) |
| Private LB | Nearly unchanged (author notes "best private score") |
| Runtime | ~65 minutes (1 hour 5 minutes) |

**Notable**: Author achieved minimal shakeup between public and private scores, validating the conservative baseline approach.

## Connections

- **Related concepts**:
  - Attention-pooled SED (similar to experiment 008)
  - Multi-model ensemble strategies
  - OpenVINO CPU optimization

- **Informs experiments**:
  - Post-processing techniques (power adjustment, temporal smoothing)
  - Ensemble weighting strategies beyond simple averaging
  - Conservative tuning to avoid LB overfitting

- **Similar approaches**:
  - References i2nfinit3y/bird2025-single-sed-model-inference-lb-0-857
  - References myso1987/post-processing-with-power-adjustment-for-low-rank
  - References hideyukizushi/bird25-weightedblend-nfnet-convnextv2-lb-860
  - Based on kurisew/lb0-855-openvino-multithread-tta-ensemble-infer
