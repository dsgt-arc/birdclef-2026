---
slug: "salmanahmedtamu/20th-place-training"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 13
---

# Kaggle Summary: 20th Place Training

**Source**: https://www.kaggle.com/code/salmanahmedtamu/20th-place-training
**Competition**: BirdCLEF 2025
**Score**: 20th place (final ranking)

## Summary

This solution uses EfficientNet-based models with extensive audio preprocessing, human voice filtering, and a Model Soup averaging technique to achieve competitive performance on the BirdCLEF 2025 challenge. The approach emphasizes robust spectrogram augmentation, intelligent sampling strategies for long recordings, and ensemble-like weight averaging from multiple training epochs.

## Approach

### Architecture

**Backbone**: EfficientNet-B0 variants (NoisyStudent and AdvProp pretrained) via TIMM library. Two architecture options implemented:
- **CNN mode**: Standard classification head (default configuration shown)
- **SED mode**: Sound Event Detection head with temporal attention pooling for framewise predictions

**Configuration details**:
- Input: 3-channel spectrograms (384×384 pixels)
- Multi-label classification: 206 target species (Aves, Amphibia, Insecta, Mammalia)
- Optional nocall class (207th class)
- Dropout: 0.5 head, 0.2 stochastic depth

### Data Pipeline

**Mel Spectrogram Parameters**:
- Sample rate: 32 kHz
- n_mels: 384 frequency bins
- n_fft: 4096 (fine-grained 7.8 Hz frequency resolution)
- Hop length: 420 samples (~10ms time resolution)
- Frequency range: 50-16000 Hz (filters infrasonic noise)
- Power spectrogram with 80dB dynamic range normalization

**Audio Preprocessing**:
- **Human voice filtering**: Uses Silero VAD to detect speech segments in CSA (citizen science audio) recordings
- Splits audio into human vs non-human portions
- For training: randomly samples either filtered non-human audio or recombines human/non-human segments
- Can inject human speech from other recordings as negative augmentation
- **RMS-based sampling**: Selects 5-second chunks with maximum root-mean-square energy from longer recordings
- Combined vs chunked sampling strategies to handle varying recording lengths

**Waveform Augmentations** (via audiomentations):
- Time shift: ±15% of duration
- Frequency shift: ±5% of mel bins
- Gaussian noise injection (std=0.01)

**Spectrogram Augmentations**:
- SpecAugment: 2 frequency masks (18 bins), 2 time masks (50 frames)
- Local/global stretch: Random time-frequency warping to simulate pitch/tempo variations
- Random erasing: 70% probability, 2-8% area, aspect ratio 0.5-2.0
- Delta features: Optionally stack first and second-order deltas for 3-channel input

**Max Trick Multi-Chunk Augmentation**:
- Splits 15-second audio into 3×5-second chunks
- Creates batch dimension per sample (B×K×C×H×W)
- Takes max prediction across chunks: simulates TTA at training time
- Optional random shuffle of chunks
- Optional "maximum mix" augmentation: element-wise max of shuffled predictions

**Label Strategy**:
- Primary labels: Binary one-hot
- Secondary labels: Also marked as 1.0 (multi-label)
- Optional upsampling for rare species (<15 samples to 15)
- Optional nocall class from FF1010Bird dataset

**Train/Val Split**:
- 5-fold StratifiedKFold on primary species
- Removes species unseen in validation fold from training (prevents overfitting to rare birds)
- Tracks unseen species separately in validation metrics

### Training

**Loss Function**: Hybrid Focal + BCE
- Binary cross-entropy with logits baseline
- Focal loss (alpha=0.25, gamma=2.0) to handle class imbalance
- Weighted combination: 1.0×BCE + 1.0×Focal

**Optimizer & Schedule**:
- AdamW: LR 1e-4, weight decay 1e-4
- Cosine annealing to 1e-7 minimum
- Gradual warmup: 3 epochs with 10× multiplier ramp-up
- Gradient clipping: max norm 10.0

**Training Configuration**:
- 15 epochs
- Batch size: 32
- Mixed precision: Optional (disabled in shown config)
- Gradient accumulation: Optional (1 step = disabled)
- Multi-GPU via DataParallel

**Regularization**:
- Mixup: alpha=1.0 on spectrograms and labels
- Stochastic depth (drop_path_rate=0.2)
- Dropout 0.5 in classifier head
- Random erasing on spectrograms

**Model Soup Technique** (epochs 12-15):
- Tracks validation metrics (cmAP@1, cmAP@5, mAP, unseen species score)
- Saves models from epochs that improve any metric
- Averages state_dicts of all improving checkpoints
- Creates final ensemble-like model from weight averaging

### Inference / Post-Processing

**Validation Metrics**:
- Competition metric: Padded cmAP@5 (adds 5 dummy all-positive rows)
- Also tracks cmAP@1, micro-averaged mAP
- Separate scoring for unseen species (not in training fold)

**Max Trick at Inference**:
- Same 3-chunk split as training
- Takes element-wise max prediction across chunks
- Provides temporal redundancy without explicit ensemble

**Model Selection**:
- Best validation loss checkpoint
- Best unseen species score checkpoint
- Model soup averaged weights from improving epochs

## Key Ideas

- **Intelligent audio filtering**: Silero VAD to detect and optionally remove human speech from citizen science recordings, improving signal-to-noise ratio for bird calls
- **RMS-based chunk selection**: Instead of random cropping, selects 5-second windows with highest energy content from long recordings
- **Max trick training augmentation**: Splits each sample into multiple temporal chunks and takes max prediction, effectively doing test-time augmentation during training
- **Model soup**: Weight averaging of checkpoints from epochs that improve validation metrics (any of 4 tracked metrics), creating ensemble-like performance from single training run
- **Fine-grained spectrograms**: 4096 FFT bins give 7.8 Hz resolution for capturing harmonic structure of insect/amphibian calls
- **Local/global stretch augmentation**: Custom time-frequency warping that applies both global resizing and localized stretching to random spectrogram regions
- **Hybrid focal-BCE loss**: Combines standard BCE with focal loss to handle extreme class imbalance (some species <10 samples)
- **Secondary label inclusion**: Uses all labeled species in recording (not just primary), increasing supervision signal
- **Unseen species tracking**: Explicitly monitors performance on species not in training fold to detect overfitting

## Results

| Metric | Value |
|--------|-------|
| CV     | Not reported |
| Public LB | Not reported |
| Private LB | 20th place |

**Notes**:
- Configuration shown targets fold-specific training (configurable via `fold` parameter)
- Model soup consistently improved over single-epoch checkpoints
- Unseen species scoring suggests good generalization beyond training distribution
- Max trick (3×5sec chunks) provides consistent gains over single 5sec sampling

## Connections

- **Related concepts**:
  - Multi-chunk temporal aggregation (similar to our max-pooling over 128×1 patches)
  - Voice activity detection for audio cleaning
  - Model weight averaging (related to SWA, FGE techniques)
- **Informs experiments**:
  - Energy-based sampling strategy could improve our long-recording handling
  - Model soup technique applicable to our SSL pretraining checkpoints
  - Hybrid loss functions for class imbalance
- **Similar approaches**:
  - RMS sampling vs our current random cropping
  - SpecAugment + geometric warping vs our simpler augmentations
  - Multi-chunk inference aggregation
