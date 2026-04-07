# Experiment 021: SED Training — Tasks

## Goal

Push single SED model from 0.906 → 0.920+ LB by applying winning solution
techniques (multi-context training, AttHead variants, backbone scaling).

**Current Best SED**: 0.909 LB (v5j-pp2 single), 0.917 LB (v5j-4fold-pp2)
**Target**: 0.920+ single, 0.925+ 4-fold

### Status: **Phase G in progress. v6l (manifold mixup) training. Phases A–F complete (head/inference not the gap).**

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

## Phase F: Gap Analysis Fixes — COMPLETE (no gains)

Motivated by `notes/2026-04-05-sed-gap-analysis.md`. Comparing our v5j (0.906)
against Nikita 1st place 2025 inference code and Salman's 0.921, we identified
gaps in our pipeline. See also Nikita's reference code and the gap analysis.

Reference code: [`birdclef-2025-nikita-1st-place-sed.py`](../../../docs/references/kaggle/birdclef-2025/birdclef-2025-nikita-1st-place-sed.py)
Gap analysis: [`2026-04-05-sed-gap-analysis.md`](../../notes/2026-04-05-sed-gap-analysis.md)

### F.1 Framewise-max inference — COMPLETE (no gain)

Tested `blend_clip=0.0, blend_frame=1.0` on v5j checkpoint. Identical LB to
the 0.75/0.25 blend (0.906). LSEHead with T=1.0 is already near hard-max, so
the aggregation method isn't the gap — it's the per-frame feature quality.

- [x] F.1a Re-export v5j ONNX with `blend_clip=0.0, blend_frame=1.0`
- [x] F.1b Submit to LB — **0.906** (identical to v5j 0.75/0.25 blend)
- [ ] ~~F.1c If it helps, test on v5j-4fold too~~ — skipped, no gain

### F.4 Competition-standard AttHead — COMPLETE (regressed)

Fixed AttHead to match Nikita/Tattaka (pre-dropout, tanh attention, probability-space
clipwise). Train with dual supervision (50% clipwise + 50% framewise max), infer
with framewise-max only. Tests whether proper AttHead improves per-frame features
over LSEHead. Config: v6k (v5j recipe, hidden_dim=1024, only head changes).

- [x] F.4a Fix AttHead in `sed_head.py` (pre-dropout, tanh, sigmoid-space)
- [x] F.4b Create v6k config (v5j + AttHead + dual 50/50 + fmax inference)
- [x] F.4c Train v6k on PACE — **val/auroc 0.871 (−0.003), cmap 0.596 (−0.029)**
- [x] F.4d ONNX export + LB — **0.888 LB (−0.018)** (first 2 subs scored 0.508 from species_mapping bug)

---

## What We've Ruled Out (Phases A–F)

| Hypothesis | Test | Result | Conclusion |
|-----------|------|--------|------------|
| Site/hour priors | Phase A (v5j-pp2) | +0.003 LB | Minor, not the gap |
| CE loss | Phase B (v6a) | −0.020 offline | Doesn't work with LSEHead |
| Waveform MixUp | Phase B (v6b) | −0.019 LB | Hurts in our setup |
| Waveform augs (noise, stretch) | Phase B (v6c) | −0.013 offline | Regressed |
| More epochs (25) | Phase B (v6d) | −0.001 offline | Flat |
| AttHead (our buggy impl) | Phase C (v6e-v6j) | −0.013 to −0.017 LB | Head hurts LB |
| Backbone scaling (HGNet-B2) | Phase C (v6h) | −0.017 LB | Not the lever |
| AttHeadMultiContext (Tattaka) | Phase D (v7a-v7c) | −0.023 to −0.075 | Wrong approach |
| Multi-context LSEHead | Phase E (v8a/v8b) | −0.010 to −0.024 LB | Offline gains don't transfer |
| Framewise-max inference | Phase F.1 (v5j-fmax) | 0.906 (identical) | Pooling method is not the gap |
| Competition-standard AttHead | Phase F.4 (v6k) | −0.018 LB | Head architecture is not the gap |
| Manifold Mixup (Mix2) | Phase G.1 (v6l) | −0.007 LB | Embedding-space mixing hurts soundscape generalization |

**Pattern analysis**: After 14 training-time recipe changes across Phases A–G
(and exps 016–019), zero have improved LB. Every gain came from input
representation or post-processing changes. Full analysis:
[`notes/2026-04-06-experiment-pattern-analysis.md`](../../notes/2026-04-06-experiment-pattern-analysis.md).

## Phase G: Manifold Mixup, Backbone Exploration, TTA — IN PROGRESS

The 0.906→0.920+ gap is NOT in the head architecture or inference strategy
(ruled out in Phases C–F). Phase G focuses on training-time augmentation
(manifold mixup), backbone diversity, and inference-time TTA.

### Rationale

1. **Manifold Mixup** (Mix2-inspired) — Moummad et al. 2024 showed that mixing
   in embedding space (after backbone, before head) yields +6.2% on rare anuran
   classes, while input-space mixup alone can hurt. Random alternation between
   input-space and embedding-space mixing per batch is the key. Directly relevant:
   our 82 species with <100 recordings and 28 soundscape-only species are exactly
   the rare-class scenario. See `docs/vendor/mix2-mixture-of-mixups.md`.

2. **Backbone exploration with LSEHead** — Phase C tested backbones but confounded
   with AttHead swap (which regresses on LB). We don't know how ConvNeXt, RegNet,
   or ECA-NFNet perform with our proven LSEHead+BCE recipe. Salman explicitly moved
   away from EfficientNet for 0.921. EliKal shows 0.906-0.909 across 4 architectures.

3. **Time-shift TTA** — inference-only augmentation (shift ±0.5-1s, average
   predictions). EliKal's "only" inference trick. We have zero TTA. Can stack on
   top of any model without retraining.

### Deprioritized (from prior "Remaining Gaps")

- **3-channel input**: mathematically near-equivalent after BatchNorm — the 3×
  channel output is just a scale factor that BN absorbs. 20 epochs of fine-tuning
  compensates for any initialization difference. Deprioritized in favor of manifold
  mixup which has stronger empirical evidence (Mix2 paper).
- **CE loss**: tested 3× (v6a, v6f, v6g), regressed every time (−0.015 to −0.024).
  Our recipe doesn't synergize with CE regardless of head pairing.
- **Multi-context / temporal**: Phase E showed best-ever offline but regressed on LB.
  Can't iterate reliably until offline-LB disconnect is understood.
- **Overlap reconstruction**: v5j-fmax showed inference strategy isn't the gap.

### G.1 Manifold Mixup (v6l) — TRAINING

Implemented manifold mixup as a private method in `sed_finetune.py`. Each batch
randomly alternates between input-space spectrogram mixup (50%) and embedding-space
manifold mixup (50%), matching Mix2's random alternation finding. Uses same
Beta(alpha, alpha) lambda sampling and per-sample mixing as existing input-space
mixup. Single-variable change from v5j — same backbone, head, loss, schedule.

Implementation: `_apply_manifold_mixup()` in `sed_finetune.py`, gated by
`manifold_mixup: true` config flag. When `manifold_mixup: false` (default),
`training_step` is identical to the original code path. All existing configs
are backward-compatible.

Reference: Mix2 repo cloned to `/tmp/Mix2/` (https://github.com/ilyassmoummad/Mix2)

- [x] G.1a Implement `_apply_manifold_mixup()` in `sed_finetune.py`
- [x] G.1b Wire `manifold_mixup` + `manifold_mixup_prob` params in `train.py`
- [x] G.1c Create v6l config (v5j + `manifold_mixup: true`, `manifold_mixup_prob: 0.5`)
- [x] G.1d Validate: AST syntax check, config loads, backward compat confirmed
- [x] G.1e Fast-dev-run passed (5 batches, no errors)
- [x] G.1f Train v6l on PACE — **val/auroc 0.874 (±0.000 vs v5j), cmap 0.630 (+0.005)**
- [x] G.1g Check training results — neutral offline, best epoch 15 (faster convergence)
- [x] G.1h ONNX export + LB — **0.899 LB (−0.007 vs v5j)**

### G.2 Backbone Exploration with LSEHead (v9b-v9d) — PLANNED

Swap backbone only, keep LSEHead + BCE + v5j recipe. Tests whether backbone
choice (not head) is the lever. Phase C's backbone tests were confounded by
also switching to AttHead.

Candidates (all available in timm, used in BirdCLEF winning solutions):
- **v9b**: ConvNeXt-Tiny (`convnext_tiny.fb_in22k_ft_in1k`, 28M, used by EliKal)
- **v9c**: RegNetY-008 (`regnety_008.pycls_in1k`, 6M, used by BirdCLEF 2024/2025 1st)
- **v9d**: ECA-NFNet-L0 (`eca_nfnet_l0.ra2_in1k`, 24M, used by BirdCLEF 2023/2025 1st)

Include `manifold_mixup: true` if G.1 shows improvement.

- [ ] G.2a Create v9b config (ConvNeXt-Tiny + LSEHead)
- [ ] G.2b Create v9c config (RegNetY-008 + LSEHead)
- [ ] G.2c Create v9d config (ECA-NFNet-L0 + LSEHead)
- [ ] G.2d Train v9b-v9d on PACE
- [ ] G.2e Check training results, select best
- [ ] G.2f ONNX export best → LB submission

### G.3 Time-Shift TTA (inference-only) — ALREADY IMPLEMENTED

**Discovery**: TTA is already enabled in `submission.py` (`ENABLE_TTA = True`).
The kernel uses ±2.5s shifts with 50/25/25 weighting, plus temporal smoothing
and TopN filtering. All v5j submissions (0.906 LB) already include TTA. The
0.906 score is not raw — it already has TTA + smoothing + TopN baked in.

- [x] G.3a Already implemented in `submission.py` (lines 707-719, 1062-1073)
- [x] G.3b All v5j/v6l submissions already ran with TTA enabled

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
| v5j-fmax (blend 0/1) | **0.906** | 0.874 | framewise hard-max = same as v5j 0.75/0.25 |
| v6k (fixed AttHead) | **0.888** | 0.871 | −0.018 LB, AttHead regresses vs LSEHead |
| v6l (manifold mixup) | **0.899** | 0.874 | **−0.007 LB**, neutral offline, embedding mixing hurts |
| Perch V18 (exp 020) | **0.925** | — | Two-kernel TF, reference |
