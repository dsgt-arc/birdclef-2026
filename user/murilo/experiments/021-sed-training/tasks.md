# Experiment 021: SED Training — Tasks

## Goal

Push single SED model from 0.906 → 0.920+ LB by applying winning solution
techniques (multi-context training, AttHead variants, backbone scaling).

**Current Best SED**: 0.909 LB (v5j-pp2 single), 0.917 LB (v5j-4fold-pp2)
**Target**: 0.920+ single, 0.925+ 4-fold

### Status: **Phase D: Multi-context training (v7a-v7c). Phase C LB submission in parallel.**

- **Branch**: `mg/021-sed-training`

---

## Config Comparison

### Phase B — Recipe ablations (COMPLETED, no gains)

| Config | Backbone | Head | Loss | MixUp | val/auroc | LB | Change from v5j |
|--------|----------|------|------|-------|-----------|-----|-----------------|
| v5j (baseline) | HGNetV2-B0 | LSEHead | BCE | spec | 0.874 | **0.906** | — |
| v6a | HGNetV2-B0 | LSEHead | CE | spec | 0.854 | — | −0.020 (regressed) |
| v6b | HGNetV2-B0 | LSEHead | BCE | wave | 0.871 | — | −0.003 |
| v6c | HGNetV2-B0 | LSEHead | BCE | wave+noise | 0.861 | — | −0.013 (regressed) |
| v6d | HGNetV2-B0 | LSEHead | BCE | spec | 0.873 | — | −0.001 (25ep, flat) |

**Lesson**: Recipe changes on LSEHead don't help. The gap is architectural (head type).

### Phase C — AttHead + backbone scaling (TRAINING COMPLETE)

All use **AttHead with dual supervision** (clip 0.5 + frame 0.5, blend 75% clip + 25% frame max).
Motivated by BirdCLEF 2023/2024/2025 winning solutions + Salman's explicit recommendation.

| Config | Backbone | Params | Head | Loss | val/auroc | val/cmap | Best Epoch | Delta vs v5j |
|--------|----------|--------|------|------|-----------|----------|------------|-------------|
| **v6h** | **HGNetV2-B2** | **12M** | AttHead | BCE | **0.876** | **0.658** | 18 | **+0.002** |
| v6i | ConvNeXt-V2 Nano | 15M | AttHead | BCE | 0.874 | **0.663** | 16 | ±0 |
| v6j | EffNet-B0 NS | 5.3M | AttHead | BCE | 0.874 | 0.626 | 15 | ±0 |
| v6e | HGNetV2-B0 | 6M | AttHead | BCE | 0.870 | 0.611 | 18 | −0.004 |
| v6g | EffNet-B3 NS | 12M | AttHead | CE | 0.859 | 0.611 | 19 | −0.015 |
| v6f | HGNetV2-B0 | 6M | AttHead | CE | 0.850 | 0.577 | 18 | −0.024 |

**Key findings**: CE loss dead (−0.015 to −0.024 everywhere). AttHead regresses on B0 (−0.004)
but works with larger backbone B2 (+0.002). Backbone scaling > head type. ConvNeXt best cmap (0.663).

---

## Phase A: Site/Hour Priors on SED — COMPLETE

- [x] A.1 Generate priors, create -pp2 weights dirs, register in exp 010
- [x] A.2 v5j-pp2 → **LB: 0.909** (+0.003)
- [x] A.3 v5j-4fold-pp2 → **LB: 0.917** (+0.004)

---

## Phase B: Training Recipe Ablations — COMPLETE (no gains)

- [x] B.1-B.3 Implementation (CE loss, waveform augs, reproducibility)
- [x] B.4 v6a (CE+LSEHead): val/auroc 0.854 (−0.020, regressed)
- [x] B.5 v6b (waveform MixUp): val/auroc 0.871 (−0.003)
- [x] B.6 v6c (waveform augs): val/auroc 0.861 (−0.013, regressed)
- [x] B.7 v6d (25 epochs): val/auroc 0.873 (−0.001, flat)

**Key finding**: v6a (CE+LSEHead) regressed because CE needs AttHead with
framewise max to work. 2024+2025 winners use CE on AttHead, not LSEHead.

---

## Phase C: AttHead + Backbone Scaling — TRAINING COMPLETE (LB pending)

Motivated by winning solution analysis. Our AttHead differs from competition-standard
(Tattaka/2023 5th): missing tanh, logit-space clipwise. This may explain underperformance.
See `sed_head.py` header for details.

- [x] C.1 Create AttHead configs (v6e-v6j) — dual supervision, 75/25 blend
- [x] C.2-C.7 Train v6e-v6j on PACE — all complete
- [x] C.8 Check training results — **v6h best (+0.002), CE dead, AttHead regresses on B0**
- [ ] C.9 ONNX export v6h, v6j → Kaggle inference timing validation
- [ ] C.10 Submit v6h, v6j to LB (no priors) — validate Phase C on LB

---

## Phase D: Multi-Context Training — IN PROGRESS

**Motivation**: hengck23 (154th) achieves **0.896 LB with CNN only** (no Perch, no PP)
using long input waves. Tom (6th) confirms "similar approach, fits on 4×5sec."
The key insight: train on longer windows (15-20s) for temporal context, predict on
center 5s. This is how BirdCLEF 2023 5th place (Tattaka) implements AttHead:
`train_period=15s, infer_period=5s`.

**Implementation**: New `AttHeadMultiContext` head with competition-standard attention
(softmax(tanh(att)), probability-space clipwise, train/infer period cropping).
Dataset already supports longer windows via `window_duration` config param.

### Phase D configs

| Config | Backbone | Head | Context | Batch | Rationale |
|--------|----------|------|---------|-------|-----------|
| **v7a** | HGNetV2-B0 | AttHeadMultiContext | **3×5s (15s)** | 32 | hengck23's approach on our best backbone |
| **v7b** | HGNetV2-B0 | AttHeadMultiContext | **4×5s (20s)** | 24 | Tom's approach (6th place) |
| **v7c** | EffNet-B0 NS | AttHeadMultiContext | **3×5s (15s)** | 32 | Boredom's claim: EffNet can reach 925+ |

All use v5j recipe base (AmplitudeToDB, OneCycleLR, per-sample MixUp, BCE, 20 epochs).
clipwise_weight=1.0 (no framewise loss, matching v5j).

### Tasks

- [x] D.1 Implement `AttHeadMultiContext` in `sed_head.py`
- [x] D.2 Wire into `train.py` (head registry + config passthrough)
- [x] D.3 Create v7a, v7b, v7c configs
- [ ] D.4 Train v7a (HGNetV2-B0, 3×5s) on PACE
- [ ] D.5 Train v7b (HGNetV2-B0, 4×5s) on PACE
- [ ] D.6 Train v7c (EffNet-B0 NS, 3×5s) on PACE
- [ ] D.7 Check training results, compare val/auroc
- [ ] D.8 ONNX export + Kaggle inference (needs 15s/20s windowed inference)
- [ ] D.9 Submit winners to LB — target ≥0.920 single

---

## Results Tracking

| Submission | LB | val/auroc | Notes |
|------------|-----|-----------|-------|
| v5j (exp 019 baseline) | **0.906** | 0.874 | HGNetV2-B0, LSEHead, BCE |
| v5j-pp2 | **0.909** | — | v5j + site/hour priors (+0.003) |
| v5j-4fold (exp 019) | **0.913** | — | 4-fold ensemble |
| **v5j-4fold-pp2** | **0.917** | — | v5j-4fold + priors (+0.004), current best SED |
| v6a (CE+LSEHead) | — | 0.854 | −0.020 offline, regressed |
| v6b (waveform MixUp) | — | 0.871 | −0.003 offline |
| v6c (waveform augs) | — | 0.861 | −0.013 offline, regressed |
| v6d (25 epochs) | — | 0.873 | −0.001 offline, flat |
| **v6h (HGNetV2-B2+AttHead)** | pending | **0.876** | **+0.002, best Phase C** |
| v6i (ConvNeXt-V2+AttHead) | pending | 0.874 | ±0, best cmap (0.663) |
| v6j (EffNet-B0 NS+AttHead) | pending | 0.874 | ±0, EffNet test |
| v6e (HGNetV2-B0+AttHead) | — | 0.870 | −0.004, AttHead regresses on B0 |
| v6f (HGNetV2-B0+AttHead+CE) | — | 0.850 | −0.024, CE dead even with AttHead |
| v6g (EffNet-B3 NS+AttHead+CE) | — | 0.859 | −0.015, EffNet MBConv + CE fails |
| v7a (HGNet-B0+MC 3×5s) | pending | pending | Multi-context 15s, hengck23 approach |
| v7b (HGNet-B0+MC 4×5s) | pending | pending | Multi-context 20s, Tom approach |
| v7c (EffNet-B0 NS+MC 3×5s) | pending | pending | Multi-context + EffNet |
| Perch V18 (exp 020) | **0.925** | — | Two-kernel TF, reference |
