---
slug: "team-kefir-1st-place-solution"
type: "writeup"
competition: "birdclef-2024"
date_captured: "2026-03-22"
---

# Kaggle Summary: BirdCLEF 2024 1st Place Solution (Team Kefir)

**Source**: https://www.kaggle.com/competitions/birdclef-2024/discussion/524549
**Competition**: BirdCLEF 2024
**Metric**: macro-averaged ROC-AUC (182 bird species)

## Summary

The winning solution uses an ensemble of EfficientNet-B0 and RegNetY-008 models trained with
CE loss (not BCE), pseudo-labeled unlabeled soundscapes, aggressive data filtering via Google
Bird Vocalization Classifier, and OpenVINO-accelerated CPU inference. The key innovations are
data curation (signal statistics filtering, classifier-based quality filtering) and the
surprising effectiveness of CE loss over BCE for this multi-label task.

## Approach

### Data Curation (Most Important Factor)

**Signal statistics filtering**: Instead of standard k-fold CV, they use fold0 plus filtering
by signal statistics `T = std + var + rms + pwr`. Only the bottom 80th percentile of T is kept,
discarding noisy/loud recordings. This was more effective than any model change.

**Google Bird Vocalization Classifier filtering**: For each chunk in `train_audio`:
- If Google classifier's max prediction doesn't match primary label → drop the chunk
- If Google classifier's max prediction matches a secondary label → relabel to that secondary
- Google classifier pseudo-labels added to training labels with coefficient 0.05

**Duplicate removal**: Duplicate audio files in train data are discarded.

**Soundscape pseudo-labeling**: Unlabeled soundscapes are labeled using an ensemble of
the Google classifier + their best train_audio-only models.

### Architecture

- **EfficientNet-B0** (ImageNet pretrained) — primary model
- **RegNetY-008** (ImageNet pretrained) — secondary model
- SEResNeXt tried but 3× slower inference for same accuracy
- ViTs worked significantly worse
- Complex SED architectures from prior years worse than pure backbones

### Training

- **Loss**: CrossEntropy (NOT BCE — CE was significantly better)
  - During training: logits → softmax (multiclass framing)
  - During inference: logits → sigmoid (for multi-label output)
  - Rationale: with 182 classes and usually only 1 present, problem reduces to multiclass
- **Optimizer**: AdamW, CosineAnnealingLR
- **LR**: 1e-3 to 3e-3
- **Epochs**: 7-12
- **Batch size**: 96
- **Input**: 10-second chunks (two adjacent 5s chunks with averaged labels)
- **Audio segmentation**: During each epoch, model sees one random chunk per file. Large files
  segmented into X-second pieces (tried 20, 30, 60s) to balance train_audio vs soundscapes.

**Augmentation**:
- Random audio segment selection
- XY masking (SpecAugment-style)
- Horizontal cutmix

**Mel spectrogram**:
- n_fft=1024, hop_length=500, n_mels=128
- fmin=40, fmax=15000, power=2

### Inference / Post-Processing

**Overlapping context**: To predict chunk n, model takes 10 seconds: 5s from chunk n + 2.5s
from previous and next chunks.

**Ensemble**: `min()` reduction across models within architecture, then `mean()` across
architectures. The `min()` reduction lowers uncertain/noisy predictions from sigmoid.

**Inference optimization**:
- OpenVINO compilation (fixed input shape)
- Parallel mel spectrogram computing with joblib
- All mel spectrograms stored in RAM

### What Didn't Work

**Negligible/no improvement**: nocall class, mel normalization variants, 224×224 input,
15s chunks, softmax temperature, SWA/EMA, pseudo-label temperature tuning, pretraining
on prior years' data.

**Decreased score**: BCE/focal loss, mixup/noise/pixdrop/blur augmentations, STFT instead
of mel, additional Xeno-Canto data (hurt both public and private), high-coefficient pseudo-
labeling of train data, train 10s/infer 5s mismatch, bird call detection models, multi-stage
training.

## Progress Table

| Step | Private | Public |
|------|---------|--------|
| Baseline (EffNet-B0, CE, softmax, fold0) | 0.544 | 0.600 |
| Sigmoid for inference | 0.588 | 0.629 |
| Random 5s chunk selection | 0.602 | 0.639 |
| XY masking | 0.602 | 0.639 |
| Horizontal cutmix | 0.615 | 0.670 |
| Pseudo-labeled unlabeled soundscapes | 0.640 | 0.687 |
| 60s segmentation | 0.650 | 0.692 |
| Secondary labels | 0.643 | 0.695 |
| Google classifier filtering | 0.655 | 0.703 |
| 10s input (context overlap) | 0.670 | 0.716 |
| Ensemble (min) 5 EffNet-B0 | 0.689 | 0.735 |
| **Final**: Ensemble 6 EffNet + 3 RegNet | **0.691** | **0.739** |

## Key Takeaways

1. **Data quality > model complexity**: Signal statistics filtering and Google classifier-based
   quality filtering were more impactful than any architecture change.
2. **CE loss > BCE loss for sparse multi-label**: When most samples have only 1 active class
   out of 182, CE loss effectively reduces to multiclass and outperforms BCE.
3. **Pseudo-labeling unlabeled soundscapes**: Major gain (+0.025 private) from leveraging
   unlabeled data with high-quality pseudo-labels.
4. **Simple models win**: Pure EfficientNet-B0 backbone outperformed complex SED architectures.
5. **min() ensemble reduction**: Lowers noisy sigmoid predictions; more effective than mean.
6. **10s context for 5s predictions**: Overlapping context from adjacent chunks improves predictions.
7. **Additional external data hurt**: Xeno-Canto data decreased both public and private scores.
