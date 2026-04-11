# Deep Research: BirdCLEF SED Pipelines 0.90 → 0.93+

**Source**: Claude Opus 4.6 Deep Research
**Date**: 2026-04-11
**Prompt**: `user/murilo/notes/deep-research-prompt.md` + full `sed-architecture-overview.md`

---

## Executive Summary

The gap between 0.90 and 0.93+ comes from iterative pseudo-labeling, the
CE-train/sigmoid-infer trick, waveform MixUp, and careful inference strategy
for window mismatch — not novel architectures. Every BirdCLEF 2024–2025 top-5
solution uses EfficientNet-family CNNs with attention-based SED heads,
pseudo-labeling cycles (2–4 rounds), and ONNX/OpenVINO inference.

---

## 1. Train-on-20s, Infer-on-5s: Four Proven Strategies

### Strategy 1: The Henkel Reshape Trick (BirdCLEF 2021 2nd place)

Train on 30s crops but reshape the mel spectrogram into **6 equal 5s parts
along the batch dimension** before the CNN backbone. The backbone never sees
more than 5s — its receptive field is inherently 5s-aware. After the backbone,
tensors are concatenated back to 30s representation for pooling and the head.

At inference, 5s clips feed directly into the backbone with no reshaping needed.

Benefits:
- Better weak-label coverage (bird likely present somewhere in 30s)
- Within-recording mixup augmentation across the 6 parts
- Multiple augmented views per sample
- **No train/inference mismatch** — backbone always processes 5s

Reused by BirdCLEF 2022 3rd place (Martynov & Uematsu).

### Strategy 2: AttBlock is Inherently Length-Agnostic

The PANNs-style AttBlock uses `torch.softmax(..., dim=-1)` over time with
**no positional encoding** — attention operates via position-independent
Conv1d(kernel_size=1). A model trained on 15s clips (~90 time steps) can
process 5s clips (~30 time steps) without modification; the softmax simply
renormalizes over whatever time steps are present.

BirdCLEF 2025 5th place trained on **30–60s crops** and inferred on 5s segments
using this property.

**Key difference from our AttHead**: Our implementation also uses Conv1d
kernel_size=1, so it should be length-agnostic too. The question is why our
AttHead still regresses on LB despite this theoretical property.

### Strategy 3: Sliding Window with Offset Aggregation

BirdCLEF 2022 7th place: shift audio ±1.5s and aggregate:
`p = 0.5 * p_center + 0.25 * p_right + 0.25 * p_left`

BirdCLEF 2024 1st place: overlapping 10s windows (5s target + 2.5s from each
neighbor), averaged.

### Strategy 4: Train on 5s, Rely on Pseudo-Labeling

BirdCLEF 2025 2nd place (private 0.928) and BirdCLEF 2024 3rd place both trained
on 5s clips. Pseudo-labeling mitigates weak-label noise instead of longer windows.

---

## 2. CE Loss + Sigmoid Inference: The +0.044 Trick

**BirdCLEF 2024 1st place**: trained with standard `nn.CrossEntropyLoss` (log-softmax
internally), switched to **sigmoid at inference** — gained **+0.044 private LB**
(0.544 → 0.588) from this single change alone. BCE, BCE with positive weights,
and focal loss all produced "significantly worse results."

### Why It Works

- **Training**: focal recordings are single-label (one species per file). CE with
  softmax forces competition between classes — the right inductive bias when only
  one species is present. Stronger gradients than BCE.
- **Inference**: soundscapes have multiple overlapping species. Softmax forces
  probabilities to sum to 1, suppressing co-occurring predictions. Sigmoid treats
  each class independently, matching the multi-label test scenario and ROC-AUC metric.

**This is standard multi-class CE with softmax, applied to the primary label only
(single ground-truth class per sample).** Not per-species binary CE.

### Compensating for CE+Sigmoid Noise

CE+sigmoid produces noisier outputs than pure BCE. BirdCLEF 2024 1st place
compensated with **min-ensemble reduction** (element-wise minimum across model
predictions rather than mean). The min operation suppresses uncertain predictions.

### Not Universal

Most other top solutions use BCE or FocalBCE throughout. BirdCLEF 2025 2nd place
used FocalBCELoss with label smoothing 0.1. The CE+sigmoid trick seems most
advantageous when training data is predominantly single-label and test is multi-label.

---

## 3. Waveform MixUp: Additive Mixing with Max Labels

Top BirdCLEF solutions overwhelmingly perform MixUp in the **raw waveform domain**
before mel spectrogram conversion. The implementation differs from standard image
MixUp:

- **Audio mixing**: Waveforms are **added** (element-wise summation), simulating
  natural sound overlay. NOT the standard `λ*x1 + (1-λ)*x2` weighted blend.
- **Label mixing**: Labels use **element-wise maximum** (union), not weighted
  average. For soft pseudo-labels, element-wise sum clipped to [0,1].

**Impact**: BirdCLEF 2025 2nd place ablation: removing MixUp dropped private
score from 0.871 to 0.835 — **−0.036, the largest single-augmentation impact.**

### Why Our Waveform MixUp May Have Failed

We used the standard image MixUp formula (`λ*x1 + (1-λ)*x2` with weighted labels).
The correct BirdCLEF approach is **additive mixing with max labels** — fundamentally
different. The mel transform is nonlinear, so mixing before vs after produces very
different spectrogram patterns. Additive waveform mixing creates more realistic
multi-species overlap.

---

## 4. Crop Selection: Beginning/End Bias Beats Energy-Based

Energy-based crop selection (selecting the loudest region) has been **explicitly
tested and rejected** by top competitors.

- BirdCLEF 2025 38th place: "I tried picking the 5 seconds with the highest
  RMS (energy)." The middle 5 seconds consistently outperformed energy-based.

**Winning strategies are simpler**:
- BirdCLEF 2025 2nd place: random crops from **first or last 7 seconds** (because
  "recordings are often started when the target animal is already vocalizing and
  stopped when it ceases")
- BirdCLEF 2024 3rd place: random crops from first 6 or last 6 seconds
- BirdCLEF 2025 4th place: first 30 seconds (60s for rare classes)

Winners address weak labels through pseudo-labeling and augmentation rather than
filtering silence. Aggressive audio curation (removing silence) actually **hurt**
performance in 2025 2nd-place experiments.

---

## 5. Data Augmentation Ranked by Competitive Impact

### Tier 1 (Critical)
- Waveform MixUp (−0.036 without it)
- Background noise mixing from ESC-50 and prior soundscapes (50% probability)
- SpecAugment time/frequency masking (freq max_length=10, time max_length=20, 30% each)
- Pretraining on prior BirdCLEF years + Xeno-Canto (2025 2nd place: 819K recordings,
  7,489 species)

### Tier 2 (Commonly Used)
- Gaussian/pink noise injection (noise_factor ~0.005)
- Time shifting (1-second window)
- Pitch shift (±4 semitones, p=0.1)
- Label smoothing (α=0.05–0.1)
- Horizontal CutMix on spectrograms (+0.013 in 2024 1st place)

### Tier 3 (Variable)
- Speed/tempo perturbation
- Gain adjustment
- Human voice injection (2025 1st place)

### Augmentations That Failed
- CutMix on waveforms (inferior to MixUp)
- Image-style flipping of spectrograms (reverses temporal meaning)
- Over-aggressive noise injection (50–60% destroyed scores)
- Volume augmentation alone (amplitude discarded in mel transform)

**Key finding**: Heavier augmentations slightly decreased public scores but
**significantly improved private scores** — more robust generalization.

---

## 6. Post-Processing Strategies

### Min-Ensemble Reduction (+0.003)
Element-wise minimum across ensemble members instead of mean. Suppresses noisy
CE+sigmoid predictions. BirdCLEF 2024 1st place.

### Temporal Prediction Smoothing (+0.01)
Adjacent 5s window predictions blended: kernel `[0.1, 0.2, 0.4, 0.2, 0.1]`.
BirdCLEF 2024 2nd/3rd/4th place.

### Overlapping Window Inference (+0.01–0.015)
Feed 10s inputs (5s target + 2.5s neighbors) and average. BirdCLEF 2024 1st/4th.

### Soundscape-Level Enhancement (+0.02 public)
If a species is detected with high confidence anywhere in a 60s file, boost it
in all windows. Marginal on private.

### Cut-Off Filtering
If no species detected with >0.10 probability in any segment of a file, halve
all probabilities. Reduces false positives in silent recordings.

---

## 7. The 0.90 → 0.93 Recipe (BirdCLEF 2025 1st Place Progression)

| Stage | Score | Key Addition |
|-------|-------|-------------|
| Baseline EfficientNet-L0 + CE | 0.872 | — |
| + pseudo-labeling round 1 + MixUp + StochasticDepth | 0.898 | +0.026 |
| + four rounds of pseudo-labeling with PowerTransform | 0.930 | +0.032 |
| + separate pipeline for insects/amphibians | 0.933 | +0.003 |

Largest gains come from **iterative pseudo-labeling** (+0.03–0.06 across 2–4
rounds) — responsible for the biggest score jumps in nearly every winning solution.

---

## 8. Implications for Our Pipeline

### What we should try (prioritized)

1. **Fix waveform MixUp implementation** — use additive mixing with max labels,
   not weighted blend. This is fundamentally different from what we tested.

2. **CE loss + sigmoid inference** — apply standard multi-class CE to primary
   label during training, switch to per-class sigmoid at inference. Our previous
   CE tests may have used the wrong formulation.

3. **Pseudo-labeling with the v5j teacher** — our exp 025 attempt used wrong
   parameters. The winning approach uses 2–4 rounds of iteration with the
   student becoming the teacher.

4. **Overlapping window inference** — feed 10s context (5s target + 2.5s
   neighbors) instead of isolated 5s segments. Simple change to Kaggle kernel.

5. **Beginning/end crop bias** — random crops from first or last 7 seconds of
   focal recordings instead of fully random.

6. **Temporal prediction smoothing** — blend adjacent segment predictions with
   a smoothing kernel. Pure inference-time change.

### What we should NOT prioritize

- Longer training windows (the mismatch issue is real, but Strategy 1 solves it)
- Novel architectures (every winner uses EfficientNet + AttBlock)
- Energy-based crop selection (explicitly rejected by competitors)
- More backbone scaling (diminishing returns)
