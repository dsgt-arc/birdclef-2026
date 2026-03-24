# Experiment 016: SED Architecture — Summary

## Overview

Replaced clip-level classification (global avg pool → linear head) with Sound Event
Detection (SED) architecture using EfficientNet-B0. Goal: close the gap to a competitor's
0.908 single-model LB score.

**Best result**: 0.823 LB (v1 + post-processing), up from 0.767 baseline (exp 012).

## What Was Built

- `src/birdclef/heads/sed_head.py` — AttHead: GeM frequency pooling → attention temporal
  aggregation → dual clipwise/framewise output
- `src/birdclef/encoders/efficientnet.py` — EfficientNet-B0 backbone via timm with
  `forward_features()` returning (B, C, F, T) feature maps
- `src/birdclef/lightning/sed_finetune.py` — SEDFinetuneModule with dual loss, spectrogram
  mixup, configurable loss type
- `src/birdclef/data/sed.py` — 10-second window loading for focal + soundscape data
- `scripts/export_efficientnet_sed.py` — ONNX export with blended sigmoid output
- Full SLURM training, export, and Kaggle submission pipeline integration (exp 010)

## What Worked

### SED architecture (+0.028 LB over baseline)

The core SED design is validated. GeM frequency pooling + attention temporal aggregation +
framewise max inference outperforms clip-level on all metrics:

| Model | Test AUROC | Test cmAP | LB |
|-------|-----------|----------|-----|
| EfficientNet-B0 SED v1 | **0.860** | **0.593** | 0.795 |
| ProtoCLR clip-level (exp 012) | 0.845 | 0.481 | 0.767 |

cmAP improved by **+0.112** — SED's framewise detection captures brief calls that
clip-level averaging misses.

### Post-processing (+0.028 LB, no retraining)

Three inference-time improvements confirmed additive gains:

| Technique | Estimated | Actual |
|-----------|-----------|--------|
| TTA (2.5s temporal shifts, 3× forward) | +0.012 | combined |
| Gaussian temporal smoothing | +0.010 | combined |
| TopN (file-level max multiplication) | +0.011 | combined |
| **Total** | **+0.033** | **+0.028** |

### 10s training windows

Training on 10s windows (vs 5s) provides temporal context. The model sees 2.5s of context
before and after the target segment, improving detection of calls that span window boundaries.

## What Didn't Work

### v2: NS-JFT pretrained + CE loss (−0.068 LB)

Changed two variables simultaneously: encoder weights (`tf_efficientnet_b0.ns_jft_in1k`)
and loss function (standard CE with label_smoothing=0.05).

| Metric | v1 (best) | v2 | Delta |
|--------|-----------|-----|-------|
| test/auroc | 0.860 | 0.837 | −0.023 |
| test/cmap | 0.593 | 0.539 | −0.054 |
| LB (+pp) | **0.823** | **0.755** | **−0.068** |

The LB regression (−0.068) was 3× worse than the offline regression (−0.023), suggesting
the changes hurt calibration and soundscape generalization.

**Root cause candidates** (confounded, cannot isolate):
1. NS-JFT features may need lower backbone lr (1e-5 vs 1e-4) to avoid destroying
   pretrained representations
2. Standard CE with argmax is wrong for multi-label audio — our v2 implementation
   used `labels.argmax(dim=-1)` which picks an arbitrary species when multiple are present
3. Label smoothing may conflict with the multi-label nature of the data

**Lesson**: Change one variable at a time.

## Remaining Gap Analysis (0.823 → 0.908)

### 1. Spectrogram mixup vs raw waveform mixup (est. +0.03-0.04)

**Status**: NOT IMPLEMENTED. This is likely our single biggest gap.

Our mixup applies in spectrogram domain (in `training_step` after mel transform).
The competitor explicitly says "Mixup on Raw Waves". The 2025 2nd place confirmed
**+0.036 Private LB** from audio-domain mixup.

**Why it matters**: Mixing raw waveforms before mel conversion preserves phase relationships.
Spectrogram-domain mixing creates unrealistic frequency patterns that don't occur in nature.

**Fix required**: Move mel transform from dataset `__getitem__` (CPU workers) into the
model's `forward()` pass (GPU), then apply mixup on raw waveforms in `training_step`.
This is a pipeline architecture change.

### 2. Loss function implementation (est. +0.01-0.02)

**Status**: NEEDS INVESTIGATION. Our loss may be suboptimal.

The competitor says "CrossEntropyLoss on clipwise and framewise max" — this is likely
standard `nn.CrossEntropyLoss` with the primary species as the target class.

Our v1 "soft CE" normalizes multi-hot labels to sum to 1:
```python
soft_targets = labels.float() / labels.sum(dim=-1, keepdim=True)
F.cross_entropy(logits, soft_targets)
```

For single-species samples (majority of focal data), this is equivalent to standard CE.
But for multi-species samples, it dilutes the gradient signal. The competitor likely uses
a simpler approach that just ignores secondary labels during training.

Our v2 "ce" mode used `labels.argmax(dim=-1)` which is broken for multi-label — it
picks the lowest-indexed species arbitrarily.

### 3. EfficientNet-B0 model variant (est. unknown)

**Status**: NEEDS INVESTIGATION. Are we using the right model?

Our v1 uses `efficientnet_b0` (standard ImageNet-RA pretrained). The competitor didn't
specify which variant. Options in timm:
- `efficientnet_b0` — what we use (ImageNet RA pretrained)
- `tf_efficientnet_b0` — TF-ported variant (slightly different)
- `tf_efficientnet_b0.ns_jft_in1k` — Noisy Student on JFT (we tried, hurt)
- `tf_efficientnet_b0_ap` — AdvProp trained

There may also be audio-pretrained EfficientNet-B0 models (PANNs, BirdNET) that provide
better initial features for spectrogram classification than ImageNet weights.

### 4. 5s inference on 10s-trained model (est. +0.01-0.02)

**Status**: NOT IMPLEMENTED.

We train on 10s windows but infer on 5s (500 frames vs 1000). The model learned temporal
patterns over 10s that are halved at inference. The competitor "trained on 10 sec"
suggesting they also inferred on 10s with overlapping windows.

**Fix**: Implement 10s overlapping inference with 5s step, reconstruct per-5s predictions.

### 5. Model soup (est. +0.005-0.01)

**Status**: NOT IMPLEMENTED.

Average last 3-5 epoch checkpoints (epochs 20-24) for free stability gains. No retraining
needed, just weight averaging and re-export.

## Score Timeline

| Version | Changes | LB | Delta |
|---------|---------|-----|-------|
| Exp 012 ProtoCLR | Clip-level baseline | 0.767 | — |
| Exp 016 v1 | SED + soft CE + spectrogram mixup | 0.795 | +0.028 |
| Exp 016 v1 + pp | + TTA + smoothing + TopN | **0.823** | +0.056 |
| Exp 016 v2 + pp | + NS-JFT + CE (regression) | 0.755 | −0.012 |

## Key Artifacts

| Artifact | Path |
|----------|------|
| Best checkpoint (v1) | `checkpoints/2026-effnet-b0-sed-v1/best-epochepoch=24.ckpt` |
| ONNX model (v1) | `010-.../artifacts/weights/effnet-b0-sed-v1/model.onnx` |
| v1 metrics | `artifacts/training/efficientnet_b0_sed/metrics.json` |
| v2 metrics | `artifacts/training/efficientnet_b0_sed_v2/metrics.json` |
| W&B v1 | `dsgt-arc/birdclef-2026-sed-architecture/runs/x5rs663d` |
| W&B v2 | `dsgt-arc/birdclef-2026-sed-architecture/runs/covhhpck` |

## Competition Context

Our approach trains EfficientNet-B0 from ImageNet weights on ~29K focal + soundscape
recordings. The top public notebooks take a fundamentally different approach:

| Approach | Best LB | Strategy |
|----------|---------|----------|
| Perch v2 + ProtoSSM + MLP | 0.917 | Frozen foundation model + lightweight heads |
| Perch v2 + ProtoSSM | 0.913 | Same family, different tuning |
| Perch v2 baseline | 0.907 | Foundation model + simple probes |
| **Competitor SED (reference)** | **0.908** | **Train-from-scratch SED (our approach)** |
| **Our SED v1 + pp** | **0.823** | **Same approach, gap in implementation** |

The foundation model approaches (Perch-based, 0.907-0.917) use Google's pretrained bird
audio model as a frozen feature extractor. They don't train on focal recordings at all.

The competitor's 0.908 proves that a train-from-scratch SED approach can match foundation
models. The gap between our 0.823 and 0.908 is implementation detail, not architecture.

## Recommendations

### Priority 1: Raw waveform mixup (highest expected ROI)
Move mel to model forward, mixup raw audio. Confirmed +0.036 by 2025 2nd place.

### Priority 2: Fix CE loss implementation
Use standard `nn.CrossEntropyLoss(logits, primary_species_index)` like the competitor.
Don't normalize multi-hot labels. Don't use argmax on multi-hot.

### Priority 3: Try audio-pretrained B0
Check if PANNs or BirdNET provides an EfficientNet-B0 checkpoint pretrained on AudioSet
or bird audio. These should provide better initial features than ImageNet.

### Priority 4: 10s overlapping inference
Infer on 10s windows with 5s step to match training resolution.

### Priority 5: Model soup
Average checkpoints from epochs 20-24 for free stability.
