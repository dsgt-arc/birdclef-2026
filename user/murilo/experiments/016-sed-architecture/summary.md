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

| Version | Changes | LB | Delta vs v1+pp |
|---------|---------|-----|-------|
| Exp 012 ProtoCLR | Clip-level baseline | 0.767 | — |
| Exp 016 v1 | SED + soft CE + spectrogram mixup | 0.795 | — |
| Exp 016 v1 + pp | + TTA + smoothing + TopN | **0.823** | **baseline** |
| Exp 016 v2 + pp | + NS-JFT + CE (2 changes, regression) | 0.755 | −0.068 |
| Exp 016 v3b + pp | + raw waveform mixup | 0.802 | −0.021 |
| Exp 016 v3a + pp | + TF-ported V1 weights | 0.813 | −0.010 |
| Exp 016 v3c + pp | + standard CE loss | pending | — |
| Exp 016 v3d | EfficientNetV2-B0 (correct architecture) | training | — |

## V3 Ablation Results

Three parallel experiments, each changing one variable from v1:

| Model | Change | test/auroc | test/cmap | LB (+pp) | Verdict |
|-------|--------|-----------|----------|----------|---------|
| **v1** | **baseline** | **0.860** | **0.593** | **0.823** | — |
| v3a | TF-ported weights | 0.857 | 0.599 | 0.813 | **Worse on LB** (−0.010) |
| v3b | raw waveform mixup | 0.862 | 0.607 | 0.802 | **Worse on LB** (−0.021) |
| v3c | standard CE loss | 0.850 | 0.552 | pending | **Hurt offline** |

**Key findings:**
- **Every V1 tweak made LB worse** — v1's recipe (ImageNet RA weights + soft CE +
  spectrogram mixup) is a robust local optimum for EfficientNet-B0 V1
- v3a (TF weights): LB 0.813 (−0.010), v3b (waveform mixup): LB 0.802 (−0.021)
- Offline metrics are **not predictive of LB** for these changes: v3b had the best
  offline metrics but the worst LB among v3a/v3b
- The gap to 0.889 is unlikely to be closed by hyperparameter tweaks on V1

**Critical discovery:** The competitor used **EfficientNetV2-B0** (V2 architecture),
not EfficientNet-B0 (V1). V2 has Fused-MBConv blocks, 5.6M params (vs V1's 3.6M),
and different channel widths. This is a fundamental architecture mismatch, not a
hyperparameter gap. v3d tests this.

## Competition Context

| Approach | Best LB | Strategy |
|----------|---------|----------|
| Pantanal Distill (Perch + ProtoSSM v3) | 0.917 | Frozen foundation model + SSM + probes |
| Perch + ProtoSSM | 0.913 | Same family, different tuning |
| Perch + LogReg probes | 0.910 | Simplest: frozen Perch + sklearn |
| Perch v2 baseline | 0.907 | Foundation model + simple probes |
| **Competitor SED (reference)** | **0.889** | **EfficientNetV2-B0 SED, single model** |
| **Our SED v1 + pp** | **0.823** | **EfficientNet-B0 (V1) SED** |

Source: [discussion/683791](https://www.kaggle.com/competitions/birdclef-2026/discussion/683791).
Note: the 0.908 figure from earlier analysis may have been from a different competitor
or an ensemble. The post author confirmed 0.889 with EfficientNetV2-B0 single model.

## Potential Next Steps (Priority Order)

### 1. EfficientNetV2-B0 (v3d — currently training)

**Status**: Training. Most promising change.

The competitor confirmed using EfficientNetV2-B0 (`tf_efficientnetv2_b0` in timm), not
our EfficientNet-B0 (V1). V2 is a fundamentally different architecture:
- Fused-MBConv blocks in early layers (faster training, better features)
- 5.6M params vs 3.6M (56% more capacity)
- Different channel widths (192 vs 320 at last stage)

If v3d shows significant improvement, combine with other winning changes (v4).

### 2. Train/inference window alignment

**Status**: NOT INVESTIGATED.

We train on 10s windows (1000 mel frames) but infer on 5s (500 frames). The model
learns temporal patterns over 10s that are halved at inference. Two approaches:

**Option A: 10s overlapping inference (inference-only change)**
- Load 10s centered on each 5s target (2.5s context before + after)
- Run inference on 10s mel, extract prediction for the center 5s
- Overlap-average predictions from adjacent windows
- Confirmed +0.015 LB by 2024 1st place
- No retraining needed, just inference pipeline change

**Option B: Train on 5s to match inference (training change)**
- The 2025 1st place trained AND inferred on 5s (same window size)
- Eliminates train/inference mismatch entirely
- Would require retraining but simplifies the pipeline
- May lose the temporal context benefit of 10s training

### 3. Mel spectrogram parameters

**Status**: NOT INVESTIGATED.

Our mel config differs significantly from the 2025 1st place:

| Parameter | Ours | 2025 1st Place |
|-----------|------|----------------|
| n_fft | 1024 | 4096 |
| n_mels | 128 | 224 |
| hop_length | 320 (10ms) | 1252 (39ms) |
| fmax | None (Nyquist) | 16,000 Hz |

Higher n_fft gives better frequency resolution (important for distinguishing similar
bird calls). More mel bins (224 vs 128) provide finer spectral detail. These changes
would require retraining since the input dimensions change.

### 4. Model soup (free, no retraining)

**Status**: NOT IMPLEMENTED.

Average checkpoints from epochs 20-25 for stability. Zero cost — just weight averaging
and re-export. Should be done for any model that goes to final submission.

### 5. Ensemble with Perch-based approach

**Status**: NOT STARTED.

The Perch-based approaches (0.907-0.917) are fundamentally different from our SED
approach — they use a frozen foundation model. Ensembling our trained SED with
Perch predictions could capture complementary strengths:
- Our SED: trained on focal recordings, learns species-specific spectral features
- Perch: pretrained on 14K+ species, strong zero-shot generalization

A simple weighted average or learned ensemble could push beyond either approach alone.

## References

- [2025 1st place inference](../../vendor/kaggle/notebooks/birdclef-2025/birdclef2025-1st-place-inference/):
  5s windows (no overlap), 224 mels, n_fft=4096, spectrogram mixup, 7-model ensemble
- [SED improvement plan](../../docs/references/kaggle/birdclef-2026/discussions/sed-improvement-plan-681146.md):
  28 SED experiments, technique rankings, code snippets
- [Competitor discussion](https://www.kaggle.com/competitions/birdclef-2026/discussion/683791):
  EfficientNetV2-B0 SED = 0.889 single model
- Kaggle notebook analyses:
  [Pantanal Distill 0.917](../../docs/references/kaggle/birdclef-2026/notebooks/pantanal-distill.md),
  [Perch+ProtoSSM 0.913](../../docs/references/kaggle/birdclef-2026/notebooks/perch-protossm-zirach.md),
  [Perch+LogReg 0.910](../../docs/references/kaggle/birdclef-2026/notebooks/perch-probes-chiranjithdharma.md),
  [Perch baseline 0.907](../../docs/references/kaggle/birdclef-2026/notebooks/perch-blamerx.md)
