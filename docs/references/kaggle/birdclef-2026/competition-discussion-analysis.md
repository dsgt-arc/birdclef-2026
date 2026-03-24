---
title: "BirdCLEF 2026 Competition Discussion Analysis"
source: "https://www.kaggle.com/competitions/birdclef-2026/discussion/681146"
date_captured: "2026-03-23"
related: ["../winning-solutions-analysis.md", "../birdclef-2025/birdclef2025-1st-place-inference.md"]
---

# BirdCLEF 2026 Competition Discussion Analysis

Intelligence gathered from active competition discussions and attached analysis
reports. Covers techniques, ablation results, and specific LB scores from
competitors currently participating in BirdCLEF 2026.

## Sources

- [Discussion post](https://www.kaggle.com/competitions/birdclef-2026/discussion/681146):
  Comprehensive SED improvement plan with attached reports
- [SED model comparison](https://storage.googleapis.com/kaggle-forum-message-attachments/3424971/39433/sed_model_comparison_20260320.html):
  28 experiments with detailed ablations
- [Perch baseline discussion](https://www.kaggle.com/competitions/birdclef-2026/discussion/681000):
  Google Perch model = 0.825 LB out-of-the-box
- Competitor comment: EfficientNet-B0 SED = 0.908 single model

---

## 1. Key Technique Rankings (by confirmed LB/validation impact)

### Tier 1: Architecture & Loss (+0.02–0.05 each)

| Technique | Impact | Evidence |
|-----------|--------|----------|
| **CE loss** (over BCE) | +0.044 LB | 2024 1st place ablation |
| **ASL loss** (over BCE) | +0.023 val | Competitor ablation: v5→v9 |
| **Audio-domain MixUp** | +0.036 Private | 2025 2nd place: 0.835→0.871 |
| **Soft KD from Perch** | +0.030 val | Competitor: v9→v28 progression |
| **Multi-round pseudo** | +0.025 val | Competitor: v28→v30 |

### Tier 2: Inference & Post-Processing (+0.01–0.02 each)

| Technique | Impact | Evidence |
|-----------|--------|----------|
| **10s context windows** | +0.015 LB | 2024 1st place ablation |
| **TopN postprocessing** | +0.011 LB | 2025 2nd place confirmed |
| **TTA: 2.5s shifts** | +0.012 LB | SED improvement plan |
| **Temporal smoothing** | +0.010 LB | Gaussian [0.1, 0.2, 0.4, 0.2, 0.1] |
| **Model soup** | Free | Average last 3-5 checkpoints |

### Tier 3: Pretraining & Data (+0.01–0.02 each)

| Technique | Impact | Evidence |
|-----------|--------|----------|
| **NS-JFT pretrained B0** | Consistent | `tf_efficientnet_b0.ns_jft_in1k` > standard ImageNet |
| **819K prior-year pretrain** | +0.014 LB | 2025 2nd place |
| **Separate non-bird pipeline** | +0.003 LB | 2025 1st place |

### What DOESN'T Work

| Technique | Impact | Reason |
|-----------|--------|--------|
| Background noise aug | −0.041 | Distribution mismatch with pseudo-labels |
| Soft secondary labels | −0.043 | Conflicting gradients from co-occurrence stats |
| Rating filter ≥3 | −0.128 | Reduces diversity too aggressively for 234 classes |
| EfficientNetV2-S | −0.043 vs B0 | ImageNet-21k features transfer poorly to spectrograms |
| B3 with standard LR | Unstable | Gradient norm instability post-warmup |

---

## 2. Competitor SED Model Progression (0.7527 → 0.8139)

One competitor published detailed ablation results from 28 experiments:

| Step | Change | SS Val AUC | Δ |
|------|--------|-----------|---|
| v5 | BCE baseline (clip+frame dual) | 0.7527 | — |
| v9 | Replace BCE with ASL (γ⁻=4, γ⁺=0, clip=0.05) | 0.7755 | +0.023 |
| v27 | Add soft KD from Perch teacher | 0.7804 | +0.005 |
| v28 | Add round-4 pseudo labels + soft KD 10× | 0.8053 | +0.025 |
| **v30** | **Multi-round pseudo r1–r5 (no bg noise)** | **0.8139** | **+0.009** |

**Key insight**: ASL outperformed BCE, but the 0.908 competitor used CE. Both are
valid alternatives to BCE — the critical thing is NOT using plain BCE.

### ASL (Asymmetric Loss) Details

```python
# ASL suppresses easy negatives in imbalanced multi-label settings
# γ⁻=4: strong focusing on hard negative examples
# γ⁺=0: no focusing on positives (standard BCE for positives)
# clip=0.05: probability clipping to prevent gradient explosion
```

---

## 3. Google Perch as Baseline & Teacher

**Perch model**: Google Bird Vocalization Classifier, updated for 2025/2026.
Predicts 203/234 BirdCLEF 2026 species (all Aves + most Amphibia).

- **Out-of-the-box LB**: 0.825 (just running Perch on test soundscapes)
- **As teacher for knowledge distillation**: +0.030 validation AUC
- **Best use**: Generate soft pseudo-labels on soundscape clips, use as training targets
- **NOT effective as**: Direct training-time augmentation (v23: −0.039)

**Implication for us**: Our exp 014 used ProtoCLR+ConvNeXt as teachers (AUROC 0.8851
on labeled validation). Perch may be a stronger teacher, especially for the 203 bird
species it covers natively. Worth investigating as an alternative or additional teacher.

---

## 4. Loss Function Comparison

| Loss | Best Reported Score | Notes |
|------|-------------------|-------|
| BCE (baseline) | 0.7527 val | Standard multi-label |
| CE + label smoothing 0.05 | +0.044 vs BCE (2024) | Requires soft target normalization for multi-label |
| **ASL (γ⁻=4)** | 0.7755 val (+0.023 vs BCE) | Best for imbalanced 234-class; no multi-label issues |
| Dual BCE (clip+frame) | 0.7485-0.7527 | Needs ClipMix or unstable |
| SoftAUCLoss | Used in later pseudo-label rounds | Directly optimizes competition metric |

**Our exp 016 uses**: Soft CE (normalized multi-hot → F.cross_entropy). The competitor
evidence suggests **ASL may be better** than both BCE and CE for this specific task.
ASL doesn't require label normalization and handles multi-label natively.

---

## 5. NS-JFT Pretraining

The competitor found `tf_efficientnet_b0.ns_jft_in1k` (Noisy Student trained on
JFT-300M) consistently outperforms standard ImageNet init for audio spectrograms.

**Our exp 016 uses**: `efficientnet_b0` (standard ImageNet RA pretrained).
**Should switch to**: `tf_efficientnet_b0.ns_jft_in1k` for better feature init.

---

## 6. Post-Processing Pipeline (+0.075 total gain)

A competitor systematically tested 1,099 post-processing methods across 53 rounds:

### Best Pipeline (0.7419 → 0.8165 OOF AUC)

```
Input (raw model logits)
    → lmax_pre (Aves classes only, α=0.1, radius=1)
    → lscale (scale=0.9)
    → SoftmaxAnchorRichness (T=0.15, thr=0.5, cp_blend=0.60)
    → cSEBBs change-point segmentation (thr=0.06, blend=0.60)
    → Output
```

### Simple High-Impact Steps (for us)

1. **TopN postprocessing** (+0.011): For each species in a segment, multiply by the
   max probability of that species across all segments in the same file.
   ```python
   file_max = preds.max(axis=0, keepdims=True)
   preds = preds * file_max
   ```

2. **Temporal smoothing** (+0.010): Gaussian kernel across consecutive segments.
   ```python
   kernel = [0.1, 0.2, 0.4, 0.2, 0.1]
   for c in range(num_classes):
       preds[:, c] = np.convolve(preds[:, c], kernel, 'same')
   ```

3. **TTA: temporal shifts** (+0.012): Run 3 predictions per segment (0, −2.5s, +2.5s).
   ```python
   shifts = [0, -CLIP_SAMPLES//2, +CLIP_SAMPLES//2]
   preds = np.mean([model(get_clip(audio, start+s)) for s in shifts], axis=0)
   ```

These three are **free inference-time improvements** requiring no retraining.

---

## 7. 2025 1st Place Detailed Breakdown

**Nikita Babych — Final Score: 0.933 Private LB**

### Score Progression
```
Baseline (B0 + RegNetY + CE):           0.872
+ Pseudo round 1 + MixUp:              0.898  (+0.026)
+ PowerTransform × 4 rounds:           0.930  (+0.032)
+ Separate insect/amphibian pipeline:   0.933  (+0.003)
```

### Core Innovation: PowerTransform Pseudo-Labels
After generating ensemble pseudo-label probabilities, apply `p^γ` (γ=2.0) to
sharpen predictions before next training round. Prevents probability collapse
across 4 iterations.

```python
gamma = 2.0
pseudo_probs = pseudo_probs ** gamma
```

### SoftAUCLoss (used in later rounds)
Directly optimizes macro ROC-AUC instead of proxy losses:
```python
# Pairwise AUC approximation
pos = pred[target > 0.5]
neg = pred[target < 0.5]
diffs = pos.unsqueeze(1) - neg.unsqueeze(0)
loss = -F.logsigmoid(diffs * margin).mean()
```

### Key Details
- 7 models: 5 self-training + 2 supervised (one specialized on insects/amphibia)
- Backbones: EfficientNet-L0, B4, B3, RegNetY-016/008, B0 (insects)
- Extra data: 5,489 XC + 17,197 insect/amphibian recordings
- Augmentation: MixUp + StochasticDepth (drop_path_rate=0.2)
- Inference: OpenVINO CPU

---

## 8. 2025 2nd Place Detailed Breakdown

**VSydorskyy & Gonçalves — Final Score: 0.928 Private LB**

### Score Progression
```
NFNet-L0 Base:                          0.868
+ Label smoothing, class balancing:     0.875  (+0.007)
+ EfficientNetV2-S (in21k):            0.889  (+0.014)
+ OOF Pseudo × 3 rounds:               0.913  (+0.024)
+ TopN + Optuna ensemble:              0.928  (+0.015)
```

### Confirmed Ablation Gains
- **Audio MixUp**: +0.036 Private LB (0.835→0.871 without vs with)
- **OOF pseudo-labeling**: +0.024 Private LB (3 rounds)
- **819K-recording pretraining**: +0.014 LB (16,607 species, 50 epochs)
- **TopN postprocessing**: +0.011 LB
- **RandomFiltering augmentation**: +~1% Private LB

### Architecture
- 15 models (3 experiments × 5 folds), Optuna-optimized weights
- GeM frequency pooling (same as 2025 1st place)
- Label smoothing α=0.05, class weighting γ=−0.5

---

## 9. Implications for Our Exp 016

### Immediate Changes (before next training run)

| Change | Expected Impact | Effort |
|--------|----------------|--------|
| Switch to `tf_efficientnet_b0.ns_jft_in1k` | Better feature init | Config change |
| Consider ASL over soft CE | +0.023 vs BCE baseline | New loss implementation |
| Raw waveform mixup (not spectrogram) | +0.036 (2025 2nd place) | Pipeline change |

### Post-Training Improvements (no retraining needed)

| Technique | Expected Impact | Effort |
|-----------|----------------|--------|
| TopN postprocessing | +0.011 | ~10 lines in submission.py |
| Temporal smoothing | +0.010 | ~5 lines in submission.py |
| Model soup (avg last 3-5 ckpts) | Free stability | Checkpoint averaging |

### Future Experiments

| Technique | Expected Impact | Effort |
|-----------|----------------|--------|
| Perch as soft label teacher | +0.030 val | New pseudo-label pipeline |
| Multi-round pseudo with PowerTransform | +0.032 | Iterative training |
| SoftAUCLoss for later rounds | Direct metric optimization | New loss |
| 819K prior-year pretraining | +0.014 LB | Data collection + pretraining |
| Separate non-bird pipeline | +0.003 | Dedicated insect/amphibian model |
| StochasticDepth | Regularization | drop_path_rate=0.2 |

### What NOT to Do

- Background noise augmentation with pseudo-labels (−0.041)
- Soft secondary labels (−0.043)
- Rating-based quality filtering (−0.128)
- EfficientNetV2-S over B0 (worse soundscape generalization)
