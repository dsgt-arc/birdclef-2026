# Proposal: EfficientNet with Unfrozen Backbone and Frame Weighting

**Status**: proposed
**Created**: 2026-03-23
**Author**: Anthony

## Hypothesis

Unfreezing the EfficientNet-B1 backbone with acoustic-index-weighted frame pooling
will improve Kaggle LB score by ≥0.10 over the frozen linear probe baseline (exp 012:
0.649 private LB), achieving ≥0.75 LB.

## Background

### What we know

| Experiment | Approach | Kaggle LB | Lesson |
|-----------|----------|-----------|--------|
| exp 012 | Frozen EfficientNet + linear head | 0.649 | Frozen features are decent |
| exp 014/015 | Frozen EfficientNet + soft labels | — | Soft labels hurt, not architecture |
| murilo/011 | LoRA fine-tuned ProtoCLR/EAT/ConvNeXt | 0.736 | Fine-tuning backbone helps a lot |
| murilo/012 | + labeled soundscapes (234 classes) | 0.767 | In-domain data helps |
| murilo/015 | + pseudo-labeled soundscapes (GAP) | -0.005 | GAP dilutes pseudo-label signal |
| 2025 winners | Unfrozen backbone + SED + pseudo-labeling | 0.85+ | Full recipe |

**The gap**: We have never unfrozen the EfficientNet backbone. This is the single
highest-impact change from the 2025 winning recipe that we haven't tried.

### Why not full SED?

Focal data has **clip-level labels only** — no per-frame annotations. Training a
full SED head (frame predictions + attention pooling) with clip labels is weakly
supervised at best. Exp 015 tried this on frozen backbone and it didn't help.

Instead, we use pre-computed **acoustic indices as frame confidence weights**. This
gives us a soft attention prior that doesn't require frame labels:

- High spectral contrast → likely contains a call → upweight
- Low spectral contrast → likely silence/noise → downweight

### Frame weighting via acoustic indices

We already have per-frame acoustic indices for all 35,549 focal recordings
(exp 010-2026, 3s windows at 1s step):

| Index | Signal | Resolution |
|-------|--------|------------|
| Spectral contrast | Call presence (tonal peaks) | 3s/1s |
| Bioacoustic index | Biological energy (2-15 kHz) | 3s/1s |
| NDSI | Biophony vs anthropophony | 3s/1s |
| Spectral flatness | Tonality (low = call) | 3s/1s |

These are resampled to match the backbone's time frames (~20 frames for 5s input
at 250ms each) and used as pooling weights. The model learns features, the indices
guide which frames contribute to the clip-level prediction.

### No dedicated nocall data

BirdCLEF 2026 has no nocall/background class. All 206 species directories contain
only positive examples. Silent frames within recordings are the only "negatives,"
identifiable via acoustic indices.

### Connections

- **Forks**: Exp 012 (model, data module, training code)
- **Uses**: Exp 010-2026 acoustic indices (pre-computed)
- **Informed by**: Exp 101 (calibration — evaluation on labeled soundscapes)
- **Feeds into**: Iterative pseudo-labeling (better teacher model)

## Method

### Approach

Fork exp 012 code. Three changes: unfreeze backbone, add differential learning
rate, replace global average pooling with index-weighted pooling.

### Architecture

```
Raw Audio (B, 160000) ← 5s @ 32kHz
    │
    ▼
  nnAudio Mel + log-normalize + SpecAugment (train)
    │ (B, 1, 256, 625)
    ▼
  EfficientNet-B1 (UNFROZEN, pretrained ImageNet)
    │ (B, 1280, 8, 20)   ← 8 freq bins, 20 time frames
    ▼
  mean over freq → (B, 1280, 20)
    │
    ├── [acoustic index weights] → (B, 20) from spectral_contrast
    │     resampled from 1s resolution to match 20 time frames
    │     normalized: w = softmax(contrast / temperature)
    │
    ▼
  weighted_mean(features, weights) → (B, 1280)
    │
    ▼
  Dropout → Linear(1280, C) → clip logits (B, C)
    │
    ▼
  BCE loss with hard labels
```

### Run matrix

| Run | Backbone | Head | Data | Purpose |
|-----|----------|------|------|---------|
| **A** | Frozen | Linear + GAP | Focal only | Reproduce exp 012 baseline on 2026 data |
| **B** | Unfrozen | Attention + GAP | Focal only | Does unfreezing + attention help? |
| **C** | Unfrozen | Attention + GAP | Focal + soundscape + mixup | Does mixup with in-domain data help? |

Sequential progression — each run adds one change. Run A establishes the
baseline, Run B tests the main hypothesis (unfreezing + attention), Run C
adds domain mixing.

### Setup

| Component | Details |
|-----------|---------|
| Data | BirdCLEF 2026 focal: 35,549 files, 206 species, 5s chunks @ 32kHz |
| Indices | Pre-computed spectral contrast/bioacoustic index per recording (exp 010-2026) |
| Split | 90/10 stratified by recording |
| Backbone | `google/efficientnet-b1` (ImageNet pretrained, 6.5M params) |
| Compute | 1× RTX 6000 (24GB), ~2-4h per run |
| Dependencies | exp 009 (data module), transformers, nnAudio, pytorch-lightning |
| Code | Fork of exp 012 |

### Procedure

**Phase 1: Implementation**

1. Fork exp 012 → exp 102 (model, cli, pyproject)
2. Remove backbone freeze, add differential LR (backbone lr/10, head lr)
3. Add index-weighted pooling option (load pre-computed indices, resample to frame grid)
4. Config for BirdCLEF 2026 data paths + index paths
5. Verify with fast-dev-run

**Phase 2: Training**

6. Run A: frozen backbone + linear head, focal only (2026 baseline)
7. Run B: unfrozen backbone + attention head, focal only
8. Run C: unfrozen backbone + attention head, focal + soundscape mixup

**Phase 3: Evaluation**

9. Evaluate all runs on focal validation (AUROC, cmAP)
10. Evaluate on 739 labeled soundscape windows (exp 101 pipeline)
11. Export best to ONNX → Kaggle submission

### Training details

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LR (head) | 1e-3 | Same as exp 012 |
| LR (backbone) | 1e-4 | 10x lower to preserve pretrained features |
| Batch size | 32 | Fits on RTX 6000 with unfrozen backbone |
| Max epochs | 50 | With early stopping patience 8 |
| Optimizer | AdamW | Weight decay 1e-4 |
| Precision | bf16-mixed | Memory efficiency |
| SpecAugment | freq_mask=20, time_mask=50, 2+2 masks | Same as exp 012 |
| Scheduler | Cosine with warmup (3 epochs) | Standard |

### Variables

- **Independent**: backbone freezing (full/partial), pooling (GAP/index-weighted)
- **Dependent**: val AUROC, val cmAP, soundscape AUROC, Kaggle LB
- **Controlled**: data (2026 focal), backbone (EfficientNet-B1), mel params, seed

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| val/auroc | Macro AUROC on focal validation |
| val/cmap | Class-mean average precision on focal validation |
| soundscape/auroc | AUROC on 739 labeled windows (exp 101) |
| kaggle/lb | Public leaderboard score |

### Baseline

| Method | Source | val/cmap | Kaggle LB |
|--------|--------|----------|-----------|
| Frozen EfficientNet + linear | Exp 012 | 0.336 | 0.649 |
| ProtoCLR LoRA (focal + soundscape) | murilo/012 | 0.481 | 0.767 |

### Success Criteria

- **Confirm if**: Kaggle LB ≥ 0.75 (matches Murilo's best single model)
- **Strong confirm**: Kaggle LB ≥ 0.80
- **Reject if**: Kaggle LB < 0.70 or val/cmap < 0.336

## Limitations

- **Single backbone**: Only EfficientNet-B1. Ensemble diversity needs other architectures.
- **Focal only, 206 species**: No soundscape data, no sonotype coverage. Intentional — baseline first.
- **Index alignment**: Acoustic indices at 1s resolution need resampling to 250ms backbone frames. Interpolation may blur the signal.
- **No SED evaluation**: Without frame labels, we can only measure clip-level metrics. Frame weighting may help clip metrics even if frame predictions are imperfect.
