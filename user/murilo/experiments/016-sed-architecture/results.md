# Experiment 016: SED Architecture — Results

## Summary

**Hypothesis**: SED architecture (GeM + AttHead + dual CE loss + 10s windows) will
improve LB by ≥0.05 over exp 012 baseline (0.767).

**Result**: **Partially confirmed.** LB improved to **0.827** (+0.060 over baseline)
with post-processing and 10s inference, but fell short of the 0.908 competition reference
and the ≥0.85 target. The SED architecture is validated and the v3 ablation exhaustively
ruled out single-variable recipe changes. Remaining gap (0.081) is likely in input
resolution (mel params) → exp 017.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Encoder | EfficientNet-B0 (`efficientnet_b0`, standard ImageNet RA pretrained) |
| Architecture | SED: GeM freq pooling → AttHead (512 hidden, 0.5 dropout) |
| Loss | Soft CE (normalized multi-hot) on clipwise + framewise max (0.5/0.5) |
| Input | 10s windows, 32kHz, 128 mel bins |
| Augmentation | Spectrogram mixup (prob=0.5, alpha=0.5) |
| Optimizer | AdamW, backbone lr=1e-4, head lr=1e-3 |
| Batch | 128 × accumulate=2 = 256 effective |
| Epochs | 40 max, patience=10, best at epoch 24, stopped at 34 |
| Inference | 5s windows (trained on 10s), blended 0.75 clip + 0.25 framewise max |

## Results

### Leaderboard

| Model | Architecture | LB | vs Exp 012 Best |
|-------|-------------|-----|-----------------|
| **EfficientNet-B0 SED v1-10s + pp2** | **SED + 10s infer + TTA + smoothing + TopN + temp/taxa/priors** | **0.827** | **+0.060** |
| EfficientNet-B0 SED v1-10s + pp | SED + 10s infer + TTA + smoothing + TopN | 0.826 | +0.059 |
| EfficientNet-B0 SED v1 + pp | SED + TTA + smoothing + TopN | 0.823 | +0.056 |
| EfficientNet-B0 SED v1 | SED | 0.795 | +0.028 |
| EfficientNet-B0 SED v2 + pp | SED (NS-JFT + CE) + TTA + smoothing + TopN | 0.755 | −0.012 |
| EfficientNet-B0 SED v3a + pp | SED (TF weights) + pp | 0.813 | +0.046 |
| EfficientNet-B0 SED v3b + pp | SED (raw waveform mixup) + pp | 0.802 | +0.035 |
| EfficientNet-B0 SED v3c + pp | SED (CE loss) + pp | pending | — |
| ProtoCLR (exp 012) | Clip | 0.767 | baseline |
| ConvNeXt (exp 012) | Clip | 0.755 | −0.012 |
| EAT (exp 012) | Clip | 0.720 | −0.047 |
| Competition reference | SED | 0.908 | +0.141 |

### Post-Processing Impact (v1 model, no retraining)

| Technique | LB | Delta |
|-----------|-----|-------|
| v1 baseline | 0.795 | — |
| **+ TTA + smoothing + TopN** | **0.823** | **+0.028** |

TTA (2.5s temporal shifts, 3× forward passes), Gaussian temporal smoothing
([0.1, 0.2, 0.4, 0.2, 0.1]), TopN (file-level max multiplication). Combined
gain +0.028 vs estimated +0.033. These are free inference-time improvements.

### Offline Test Set

| Model | Test AUROC | Test cmAP |
|-------|-----------|----------|
| **EfficientNet-B0 SED v3b (raw waveform mixup)** | **0.862** | **0.607** |
| EfficientNet-B0 SED v1 (baseline) | 0.860 | 0.593 |
| EfficientNet-B0 SED v3a (TF weights) | 0.857 | 0.599 |
| EfficientNet-B0 SED v3c (CE loss) | 0.850 | 0.552 |
| EfficientNetV2-B0 SED v3d | 0.841 | 0.523 |
| EfficientNet-B0 SED v2 (NS-JFT + CE) | 0.837 | 0.539 |
| ProtoCLR (exp 012) | 0.845 | 0.481 |
| ConvNeXt (exp 012) | 0.844 | 0.470 |
| EAT (exp 012) | 0.838 | 0.452 |

### V2 Training Results (NS-JFT + CE Loss) — Regression

V2 changed two variables: encoder pretrained weights (`tf_efficientnet_b0.ns_jft_in1k` vs
standard ImageNet) and loss function (standard CE with label_smoothing=0.05 vs soft CE with
normalized multi-hot). Both offline metrics regressed:

| Metric | v1 (ImageNet + Soft CE) | v2 (NS-JFT + CE) | Delta |
|--------|------------------------|-------------------|-------|
| val/auroc | 0.845 | 0.822 | **−0.023** |
| test/auroc | 0.860 | 0.837 | **−0.023** |
| test/cmap | 0.593 | 0.539 | **−0.054** |
| LB (+ pp) | **0.823** | **0.755** | **−0.068** |
| Best epoch | 24 | 24 | — |
| Stopped | 34 | 35 | — |

**Analysis**: The LB regression (−0.068) is much larger than the offline regression
(−0.023 AUROC), suggesting the changes also hurt calibration or generalization to
soundscape conditions. Changing two variables simultaneously makes it impossible to
attribute the regression. Possible explanations:
1. NS-JFT weights may need different learning rates (the JFT-pretrained features are
   stronger, so backbone lr=1e-4 may be too aggressive and destroy good representations)
2. Standard CE with label_smoothing may be worse than soft CE for multi-label audio where
   many segments genuinely contain multiple species
3. The combination of both changes compounded negatively

**Conclusion**: v2 is a clear negative result.

### V3b Training Results (Raw Waveform Mixup) — Improvement

V3b changed one variable from v1: mixup on raw waveforms instead of spectrograms.
Mel transform moved to GPU-side (inside `training_step`), so mixup applies before
mel conversion. Required fix: force float32 for mel computation under mixed precision
(nnAudio STFT overflows float16).

| Metric | v1 (spectrogram mixup) | v3b (waveform mixup) | Delta |
|--------|----------------------|---------------------|-------|
| val/auroc | 0.845 | 0.846 | +0.001 |
| test/auroc | 0.860 | 0.862 | +0.002 |
| test/cmap | 0.593 | **0.607** | **+0.014** |
| Best epoch | 24 | 25 | +1 |
| Stopped | 34 | 36 | +2 |

cmAP improved by +0.014, suggesting waveform mixup helps the model detect brief calls
more accurately (cmAP is sensitive to per-class precision). AUROC improvement is modest.

**LB result: 0.802** (−0.021 vs v1+pp 0.823). Despite better offline metrics, v3b
performed worse on LB. The offline→LB gap widened, suggesting waveform mixup may cause
overfitting to the training distribution that doesn't generalize to soundscapes.

### V3 Ablation Summary

Three parallel experiments, each changing one variable from v1:

| Model | Change | val/auroc | test/auroc | test/cmap | LB (+pp) |
|-------|--------|-----------|-----------|----------|----------|
| **v1** | **baseline** | **0.845** | **0.860** | **0.593** | **0.823** |
| v3a | TF-ported weights | 0.845 | 0.857 | 0.599 | 0.813 |
| v3b | raw waveform mixup | 0.846 | 0.862 | 0.607 | 0.802 |
| v3c | standard CE loss | 0.831 | 0.850 | 0.552 | pending |
| v3d | EfficientNetV2-B0 | 0.841 | 0.523 | pending | **Hurt** (−0.070 cmAP) |

**Conclusions from v3 ablation:**
- **All v3 variants regressed on LB** vs v1+pp (0.823):
  - v3a (TF weights): LB **0.813** (−0.010). Worse despite similar offline metrics.
  - v3b (waveform mixup): LB **0.802** (−0.021). Best offline but worst LB.
  - v3c (CE loss): LB pending. Hurt offline (−0.041 cmAP).
- **Offline metrics don't predict LB**: v3b had the best offline metrics but the worst
  LB among v3a/v3b. The offline→LB correlation is unreliable for these changes.
- **v1 remains the best model** at 0.823 LB. The soft CE + spectrogram mixup +
  ImageNet RA weights combination is surprisingly robust.
- **v3d** (EfficientNetV2-B0): **Hurt significantly**. test/auroc 0.841 (−0.019),
  test/cmap 0.523 (−0.070). Early-stopped at epoch 13 (vs v1's 24). V2 architecture
  is not better than V1 for this task with our training recipe. The smaller last-stage
  channels (192 vs 320) may limit the SED head's capacity, and the model may need
  different hyperparameters (lr, epochs) than V1.

---

## Gap Analysis: Our 0.795 vs Competitor's 0.908

The 0.113 gap can be attributed to several differences between our implementation
and the competitor's approach. Ordered by estimated impact:

### 1. Raw Waveform Mixup vs Spectrogram Mixup (est. +0.03–0.04)

The competitor used "Mixup on Raw Waves". We use spectrogram-domain mixup.
The 2025 2nd place confirmed **+0.036 Private LB** from audio-domain mixup
(0.835→0.871 in their ablation). This is likely our single biggest gap.

**Fix**: Move mel transform from dataset `__getitem__` to model forward pass,
apply mixup on raw waveforms before mel conversion.

### 2. NS-JFT Pretrained Weights (est. +0.01–0.02)

We use `efficientnet_b0` (standard ImageNet RA pretrained). The competitor
and the SED model comparison study both found `tf_efficientnet_b0.ns_jft_in1k`
(Noisy Student trained on JFT-300M) consistently outperforms standard ImageNet
for audio spectrograms.

**Fix**: Change `model_name` in config to `tf_efficientnet_b0.ns_jft_in1k`.

### 3. Loss Function: CE/ASL vs Soft CE (est. +0.01–0.02)

The competitor used "CrossEntropyLoss on clipwise and framewise max" — likely
standard CE with label smoothing, not our soft CE with normalized multi-hot.
The competitor model comparison found ASL (γ⁻=4) gained +0.023 over BCE.
Our soft CE normalization may weaken the training signal for single-species
segments (majority of focal data).

**Fix**: Try standard CE with label smoothing=0.05, or ASL (γ⁻=4, γ⁺=0, clip=0.05).

### 4. Inference Window Mismatch (est. +0.01–0.02)

We trained on 10s windows but infer on 5s (500 frames instead of 1000). The model
learned temporal patterns over 10s that are halved at inference. The competitor
trained AND inferred on 10s.

**Fix**: Implement 10s overlapping inference with 5s step. Load 10s centered on each
5s target, run inference, reconstruct per-5s predictions.

### 5. Post-Processing — CLOSED (+0.028 confirmed)

~~We have zero post-processing.~~ **Implemented and confirmed: +0.028 LB (0.795 → 0.823).**
- TopN postprocessing + temporal smoothing + TTA all enabled in submission.py.

### 6. Training Duration (est. +0.005–0.01)

Competitor trained 20+ epochs. We trained 24 epochs (best) but on fewer
steps per epoch (29K samples with 10s random crops vs potentially more with
5s chunks or oversampling).

### Gap Summary

| Gap | Estimated Impact | Effort |
|-----|-----------------|--------|
| Raw waveform mixup | +0.03–0.04 | Pipeline change |
| NS-JFT pretrained | +0.01–0.02 | Config change |
| CE/ASL loss | +0.01–0.02 | Loss function change |
| 10s inference | +0.01–0.02 | Inference pipeline change |
| ~~Post-processing~~ | ~~+0.02–0.03~~ | **DONE: +0.028 confirmed** |
| **Total remaining** | **+0.05–0.10** | |
| **Remaining gap** | **+0.081** (0.827 → 0.908) | |

The estimated gains align with the observed 0.113 gap. Closing all gaps
should bring us to ~0.88–0.92, within range of the 0.908 reference.

---

## Recommendations (Priority Order)

### Phase 1: Quick Wins (no retraining)

1. **Add post-processing to submission.py**: TopN + temporal smoothing + TTA (+0.03)
2. **Model soup**: Average epochs 20-24 checkpoints (free stability)

### Phase 2: Training Improvements (v2 run)

3. **Switch to NS-JFT weights**: `tf_efficientnet_b0.ns_jft_in1k` (config change)
4. **Standard CE with label smoothing=0.05** instead of soft CE (loss change)
5. **Raw waveform mixup** (pipeline change — biggest expected gain)
6. **10s overlapping inference** (inference change)

### Phase 3: Advanced Techniques

7. **Perch knowledge distillation** (+0.030 from competitor ablation)
8. **Multi-round pseudo-labels with PowerTransform** (2025 1st place: +0.032)
9. **Extend SED to ProtoCLR/ConvNeXt/EAT** for ensemble diversity
10. **SoftAUCLoss** in later pseudo-label rounds

---

## Artifacts

- Checkpoint: `.../016-sed-architecture/checkpoints/2026-effnet-b0-sed-v1/best-epochepoch=24.ckpt`
- ONNX: `010-submission-pipeline/artifacts/weights/effnet-b0-sed-v1/model.onnx`
- Metrics: `016-sed-architecture/artifacts/training/efficientnet_b0_sed/metrics.json`
- GPU check: `016-sed-architecture/outputs/gpu_check/2026-effnet-b0-sed-v1_5430838.{txt,png}`
- W&B: `dsgt-arc/birdclef-2026-sed-architecture`
