# Experiment 021: SED Training — Tasks

## Goal

Push single SED model from 0.906 → 0.920+ LB by applying winning solution
techniques (multi-context training, AttHead variants, backbone scaling).

**Current Best SED**: 0.909 LB (v5j-pp2 single), 0.917 LB (v5j-4fold-pp2)
**Target**: 0.920+ single, 0.925+ 4-fold

### Status: **Phase E complete. v8b 0.896 LB (best multi-context). Center-chunk inference negligible (+0.001).**

- **Branch**: `mg/021-sed-training`
- **Architecture reference**: [`notes/sed-architecture-overview.md`](../../notes/sed-architecture-overview.md)

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

## Phase C: AttHead + Backbone Scaling — COMPLETE (LB regressed)

Our AttHead (logit-space clipwise, no tanh, 75/25 blend) differs from competition-standard
(Tattaka/2023 5th). Both LB submissions regressed vs v5j. See `atthead-comparison.md`.

- [x] C.1 Create AttHead configs (v6e-v6j) — dual supervision, 75/25 blend
- [x] C.2-C.7 Train v6e-v6j on PACE — all complete
- [x] C.8 Check training results — **v6h best (+0.002), CE dead, AttHead regresses on B0**
- [x] C.9 ONNX export v6h, v6j → Kaggle submission
- [x] C.10 Submit v6h, v6j to LB — **v6j 0.893 (−0.013), v6h 0.889 (−0.017). Both regressed.**

---

## Phase D: Multi-Context Training — COMPLETE (massive regression)

AttHeadMultiContext with 15-20s windows. All models regressed badly.

| Config | Backbone | Head | Context | val/auroc | Best Epoch | Delta vs v5j |
|--------|----------|------|---------|-----------|------------|-------------|
| v7c | EffNet-B0 NS | AttHeadMC | 3×5s (15s) | 0.851 | 19 | −0.023 |
| v7a | HGNetV2-B0 | AttHeadMC | 3×5s (15s) | 0.820 | 16 | −0.054 |
| v7b | HGNetV2-B0 | AttHeadMC | 4×5s (20s) | 0.799 | 16 | −0.075 |

**Root cause**: We implemented Tattaka's approach (AttHead + aggregate over all 15s → 1
prediction). hengck23's actual approach (from image) is different: **backbone sees 15s,
then SPLIT temporal features into 3 chunks, apply LSEHead per-chunk → 3 predictions**.
The context comes from the backbone's receptive field, not the head. We also changed
head type + window + batch size simultaneously, making it impossible to isolate variables.
See `atthead-comparison.md` for full analysis.

- [x] D.1 Implement `AttHeadMultiContext` in `sed_head.py`
- [x] D.2 Wire into `train.py` (head registry + config passthrough)
- [x] D.3 Create v7a, v7b, v7c configs
- [x] D.4 Train v7a (HGNetV2-B0, 3×5s) — **val/auroc 0.820 (−0.054)**
- [x] D.5 Train v7b (HGNetV2-B0, 4×5s) — **val/auroc 0.799 (−0.075)**
- [x] D.6 Train v7c (EffNet-B0 NS, 3×5s) — **val/auroc 0.851 (−0.023)**
- [x] D.7 Check training results — **all regressed massively, skip LB submission**

---

## Phase E: LSEHead + Temporal Split — PLANNED

**Motivation**: hengck23's image (`notes/figures/hengck23-cnn-birdclef.png`) reveals
the correct multi-context approach: backbone processes 15s spectrogram (conv receptive
field provides cross-window context), then temporal features are SPLIT into 3 chunks,
LSEHead applied per-chunk → 3 predictions. This is fundamentally different from Phase D
(which aggregated attention over all 15s). hengck23: 0.876 (5s) → **0.896 (3×5s) = +0.020**.

**Key difference from Phase D**: Only ONE variable changes from v5j — the window length.
Same LSEHead, same loss, same recipe. The head processes 5s chunks (not 15s).

### Implementation plan

1. Add `n_chunks` param to LSEHead — when >1, split framewise features into N chunks
   before LSE pooling, return (B, N, 234) clipwise
2. Modify `SEDFinetuneModule` — handle (B, N, 234) clipwise, sum BCE loss over N chunks
3. Labels: use same label for all chunks (correct for focal, approximate for soundscapes)
4. Configs: v8a (HGNet-B0, 3×5s=15s), v8b (HGNet-B0, 4×5s=20s)

### Tasks

- [x] E.1 Add `n_chunks` to LSEHead
- [x] E.2 Update SEDFinetuneModule to handle multi-chunk output
- [x] E.3 Create v8a config (HGNet-B0 + LSEHead + 3×5s)
- [x] E.4 Create v8b config (HGNet-B0 + LSEHead + 4×5s)
- [x] E.5 GPU check v8a — passed, 98% utilization, 1.1% memory
- [x] E.6 Train v8a on PACE — **val/auroc 0.882 (+0.008), cmap 0.719 (best ever)**
- [x] E.6b Train v8b on PACE — **val/auroc 0.881 (+0.007), cmap 0.727 (new best cmap)**
- [x] E.7 Check training results — both v8a/v8b show best offline metrics ever
- [x] E.8 ONNX export v8a/v8b — initial: avg all chunks. v8a LB 0.882 (−0.024), v8b LB **0.896** (−0.010)
- [x] E.9 Fix ONNX wrapper: center-chunk inference instead of averaging
- [x] E.10 Re-submit v8a with center-chunk — **LB 0.883** (+0.001 vs avg, negligible)
- [ ] E.11 Re-submit v8b with center-chunk — skipped, center-chunk didn't help v8a

---

## Future: Mix2 Augmentation (reference for potential Phase F)

**Paper**: "Mixture of Mixups for Multi-label Classification of Rare Anuran Sounds"
([arxiv 2403.09598](https://arxiv.org/abs/2403.09598)) —
see `docs/vendor/mix2-mixture-of-mixups.md` for full analysis.

Randomly alternates between 3 mixing strategies per training step:
- **Mixup** (25%): standard spectrogram-space interpolation (what we do now)
- **Manifold Mixup** (50%): mix embeddings after backbone, before head
- **MultiMix** (25%): batch-level Dirichlet interpolation in embedding space

Key result: individual methods underperform baseline, but random combination yields
+4.2% macro F-score overall, **+6.2% on rare classes** (anuran classification).
Particularly relevant for our 25 insect sonotypes and rare amphibians.

**Implementation priority**: After Phase E (multi-context). Manifold Mixup alone
(mix backbone features before LSEHead) is the highest-value single addition.
Could stack with multi-context — mix the 15s feature maps before splitting into chunks.

---

## Results Tracking

| Submission | LB | val/auroc | Notes |
|------------|-----|-----------|-------|
| v5j (exp 019 baseline) | **0.906** | 0.874 | HGNetV2-B0, LSEHead, BCE |
| v5j-pp2 | **0.909** | — | v5j + site/hour priors (+0.003) |
| v5j-4fold (exp 019) | **0.913** | — | 4-fold ensemble |
| **v5j-4fold-pp2** | **0.917** | — | v5j-4fold + priors (+0.004), current best SED |
| v6a (CE+LSEHead) | — | 0.854 | −0.020 offline, regressed |
| v6b (waveform MixUp) | **0.887** | 0.871 | **−0.019 LB**, waveform MixUp hurts |
| v6c (waveform augs) | — | 0.861 | −0.013 offline, regressed |
| v6d (25 epochs) | — | 0.873 | −0.001 offline, flat |
| v6h (HGNetV2-B2+AttHead) | **0.889** | 0.876 | **−0.017 LB**, best offline but worst LB |
| v6j (EffNet-B0 NS+AttHead) | **0.893** | 0.874 | **−0.013 LB**, EffNet beats HGNet on LB |
| v6i (ConvNeXt-V2+AttHead) | — | 0.874 | skip, AttHead regresses on LB |
| v6e (HGNetV2-B0+AttHead) | — | 0.870 | −0.004, AttHead regresses on B0 |
| v6f (HGNetV2-B0+AttHead+CE) | — | 0.850 | −0.024, CE dead even with AttHead |
| v6g (EffNet-B3 NS+AttHead+CE) | — | 0.859 | −0.015, EffNet MBConv + CE fails |
| v7a (HGNet-B0+MC 3×5s) | — | 0.820 | **−0.054**, massive regression |
| v7b (HGNet-B0+MC 4×5s) | — | 0.799 | **−0.075**, longer context = worse |
| v7c (EffNet-B0 NS+MC 3×5s) | — | 0.851 | **−0.023**, least bad but still regressed |
| v8a (HGNet-B0+LSE MC 3×5s, avg) | **0.882** | **0.882** | −0.024 LB, avg all 3 chunks dilutes |
| **v8b (HGNet-B0+LSE MC 4×5s, avg)** | **0.896** | **0.881** | **−0.010 LB**, 20s context helps |
| v8a (center-chunk inference) | **0.883** | 0.882 | center chunk only (+0.001, negligible) |
| Perch V18 (exp 020) | **0.925** | — | Two-kernel TF, reference |
