# Proposal: EfficientNet SED with Unfrozen Backbone

**Status**: proposed
**Created**: 2026-03-23
**Author**: Anthony

## Hypothesis

Unfreezing the EfficientNet-B1 backbone and adding a SED (Sound Event Detection)
head with attention pooling will improve Kaggle LB score by ≥0.10 over the frozen
linear probe baseline (exp 012: 0.649 private LB), achieving ≥0.75 LB.

## Background

### What we know

| Experiment | Approach | Result | Lesson |
|-----------|----------|--------|--------|
| exp 012 | Frozen EfficientNet + linear head | 0.649 Kaggle | Frozen features are decent |
| exp 014/015 | Frozen EfficientNet + soft labels | 0.237 cmAP | Soft labels hurt, not architecture |
| murilo/011 | LoRA fine-tuned ProtoCLR/EAT/ConvNeXt | 0.736 LB | Fine-tuning helps a lot |
| murilo/012 | + labeled soundscapes (234 classes) | 0.767 LB | In-domain data helps |
| murilo/015 | + pseudo-labeled soundscapes | -0.005 LB | GAP bottleneck limits pseudo-label gains |
| 2025 winners | Unfrozen backbone + SED + pseudo-labeling | 0.85+ | SED + unfreezing is the proven recipe |

The single biggest gap: **we have never unfrozen the EfficientNet backbone with SED**.
Murilo's models use LoRA on different backbones. His exp 015 diagnosis points to
global average pooling as the bottleneck — SED with attention pooling addresses this
directly.

### SED architecture (from 2025 winners)

```
waveform → mel spectrogram → EfficientNet backbone → (B, C, freq, time)
  → mean over freq → (B, C, time_frames)
  → frame_head: Linear(C, num_classes) → per-frame logits (B, T, num_classes)
  → att_head: Linear(C, num_classes) → attention weights (B, T, num_classes)

Training:  clip_logits = sum(sigmoid(att) * sigmoid(frame), dim=T) / sum(sigmoid(att), dim=T)
Inference: clip_logits = max(sigmoid(frame), dim=T)
```

Attention pooling during training lets the model learn which frames matter.
Framewise max during inference catches brief calls that get diluted by averaging.

### Connections

- **Forks**: Exp 012 (model, data module, training code)
- **Informed by**: Exp 101 (calibration — provides evaluation on labeled soundscapes)
- **Feeds into**: Iterative pseudo-labeling pipeline (better teacher model)
- **Priority #1** on STATUS.md

## Method

### Approach

Fork exp 012 code, make three changes:
1. Remove backbone freeze (unfreeze all or last N blocks)
2. Replace linear head with SED head (frame + attention heads)
3. Train on BirdCLEF 2026 focal data with hard labels

### Architecture

```
Raw Audio (B, 160000) ← 5s @ 32kHz
    │
    ▼
  nnAudio Mel + log-normalize + SpecAugment (train)
    │ (B, 1, 256, 625)
    ▼
  EfficientNet-B1 (unfrozen, pretrained ImageNet)
    │ (B, 1280, 8, 20)
    ▼
  mean over freq dim → (B, 1280, 20)  ← 20 time frames = 5 × 250ms
    │
    ├──► frame_head: Dropout + Linear(1280, C) → (B, 20, C)
    │
    └──► att_head: Linear(1280, C) → (B, 20, C)
    │
    ▼
  Training:  attention-weighted mean of frame logits → clip logits (B, C)
  Inference: max of frame logits → clip logits (B, C)
```

### Setup

| Component | Details |
|-----------|---------|
| Data | BirdCLEF 2026 focal: 35,549 files, 206 species, 5s chunks @ 32kHz |
| Split | 90/10 stratified by recording (same as exp 012) |
| Backbone | `google/efficientnet-b1` (ImageNet pretrained, 6.5M params) |
| Compute | 1× RTX 6000 (24GB), ~2-4h per run |
| Dependencies | exp 009 (data module), transformers, nnAudio, pytorch-lightning |
| Code | Fork of `user/anthony/experiments/012-efficientnet-linear-probe/` |

### Procedure

**Phase 1: Implementation**

1. Fork exp 012 model.py → add SED head (frame + attention)
2. Add train/inference mode switching (attention pool vs framewise max)
3. Add differential LR: backbone at lr/10, head at lr
4. Config for BirdCLEF 2026 data paths

**Phase 2: Training runs**

5. Run A: Unfreeze all + SED head (baseline)
6. Run B: Unfreeze last 3 blocks + SED head (partial unfreeze)
7. Run C: Unfreeze all + linear head (isolate SED vs unfreezing effect)

**Phase 3: Evaluation**

8. Evaluate on held-out focal validation (AUROC, cmAP)
9. Evaluate on 739 labeled soundscape windows (using exp 101 pipeline)
10. Export best model to ONNX → Kaggle submission

### Variables

- **Independent**: backbone freezing (frozen/partial/full), head type (linear/SED)
- **Dependent**: val AUROC, val cmAP, soundscape AUROC, Kaggle LB
- **Controlled**: data (2026 focal), backbone (EfficientNet-B1), mel preprocessing, seed

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| val/auroc | Macro AUROC on focal validation split |
| val/cmap | Class-mean average precision on focal validation |
| soundscape/auroc | AUROC on 739 labeled soundscape windows (exp 101) |
| kaggle/lb | Public leaderboard score |

### Baseline

| Method | Source | val/cmap | Kaggle LB |
|--------|--------|----------|-----------|
| Frozen EfficientNet + linear | Exp 012 | 0.336 | 0.649 |
| ProtoCLR LoRA (focal only) | murilo/011 | 0.531 | 0.736 |
| ProtoCLR LoRA (+ soundscapes) | murilo/012 | 0.481 | 0.767 |

### Success Criteria

- **Confirm if**: Kaggle LB ≥ 0.75 (matches or beats Murilo's best single model)
- **Strong confirm**: Kaggle LB ≥ 0.80 (enters competitive territory)
- **Reject if**: Kaggle LB < 0.70 or val/cmap < exp 012 baseline (0.336)

## Limitations

- **Single backbone**: Only testing EfficientNet-B1. Murilo's results suggest
  backbone diversity matters for ensembling.
- **Focal only**: No soundscape data in training. This is intentional — establish
  the unfrozen SED baseline before adding pseudo-labels.
- **206 species**: Focal data only covers 206/234 species. The 28 soundscape-only
  species need separate handling (see exp 101).
- **No pseudo-labels**: Murilo's exp 015 showed pseudo-labels don't help with GAP.
  SED should fix this, but we test SED alone first before combining.
