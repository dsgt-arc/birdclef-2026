# Gap Analysis: HGNetV2-B0 (0.884) vs EfficientNet-B0 SED v4b (0.860)

Date: 2026-03-30

## Summary

ttahara's HGNetV2-B0 baseline scores **0.884 LB** vs our best EfficientNet-B0
SED v4b at **0.860 LB** (0.859 raw, 0.860 with pp2). That's a **+0.024 gap**
from a single model with simpler architecture. Understanding what drives this gap
could lift our SED pipeline significantly.

## Side-by-Side Comparison

| Aspect | Our v4b (0.860) | ttahara HGNet (0.884) | Delta |
|--------|-----------------|----------------------|-------|
| **Backbone** | EfficientNet-B0 (ImageNet) | HGNetV2-B0 (SSLD stage2 ft) | Different arch + pretrain |
| **Params** | ~5.3M | ~6.0M | Similar |
| **Head** | AttHead (GeM + attention) | LSEHead (mean freq + LSE time) | Different pooling |
| **Loss** | Soft CE (dual clip+frame) | BCEWithLogitsLoss | Different loss |
| **n_fft** | 4096 | 2048 | Lower freq resolution |
| **n_mels** | 224 | 256 | +32 mel bands |
| **hop_length** | 320 (10ms) | 313 (~10ms) | ~Same |
| **f_max** | None (16kHz) | None (16kHz) | Same |
| **Mel norm** | AmplitudeToDB + standardize | AmplitudeToDB + resize + min-max | Different |
| **Input shape** | (1, 224, 1000) | (1, 256, 256) — resized | Fixed square vs raw |
| **Segment** | 10s | 5s | Half the duration |
| **Batch size** | 64 (eff. 256 with grad accum) | 64 | 4× effective difference |
| **LR** | 1e-4 encoder / 1e-3 head | 5e-4 uniform | Higher + uniform |
| **Weight decay** | 0.01 | 1e-4 | 100× less |
| **Scheduler** | CosineAnnealing + warmup | OneCycleLR (warmup built-in) | Similar shape |
| **Epochs** | 50 (early stop ~30) | 20 | Fewer |
| **MixUp** | Spectrogram (p=0.5, α=0.5) | Spectrogram (α=1.0, θ=0.8) | Stronger + label clip |
| **MixUp warmup** | None | 5 epochs | HGNet skips early |
| **CV folds** | None (90/10 split) | 4-fold MultiLabel Stratified | Ensemble benefit |
| **Soundscape data** | Included (fixed 5s windows) | Included (variable-length splits) | Different segmentation |
| **Secondary labels** | Yes (multi-hot) | Yes (multi-hot) | Same |
| **Inference** | ONNX (single model) | OpenVINO (4-fold ensemble) | Ensemble + faster runtime |
| **TTA** | ±2.5s shifts | None | We have TTA, they don't |
| **Post-processing** | pp2 (temp + taxa smooth) | None | We have PP, they don't |

## Key Differences Driving the Gap

### 1. 4-Fold Ensemble (~+0.010-0.015)
**This is likely the biggest single factor.** ttahara trains 4 models and averages
predictions. Ensemble of diverse folds reduces variance significantly. Our single
90/10 split model has no ensemble benefit.

**Actionable**: Train 4-fold CV on PACE, export all 4 ONNX models, average at
inference. OpenVINO would be fastest but ONNX also works.

### 2. LSEHead vs AttHead (~+0.003-0.008)
LSE pooling is a smooth max approximation that naturally handles sparse events.
Our attention head has more parameters but may overfit on the small dataset.
The LSE approach is parameter-efficient (just two Linear layers + LSE pool).

**Key insight**: ttahara's head does `mean(freq)` then `LSE(time)` — frequency
is averaged (all freq bands contribute equally), time is LSE-pooled (peaks
dominate). Our GeM + attention does learnable freq pooling then attention-weighted
time pooling. The simpler approach may generalize better.

**Actionable**: Implement LSEHead as an alternative to AttHead. Easy to add to
our existing SED pipeline.

### 3. Stronger MixUp (α=1.0 vs α=0.5, with θ=0.8 clipping) (~+0.002-0.005)
ttahara uses `Beta(1.0, 1.0)` = Uniform[0,1] for lambda (after max enforcement).
Our `Beta(0.5, 0.5)` is bimodal — concentrates near 0 and 1, meaning most
"mixes" are barely mixed. Uniform gives much more aggressive mixing.

The **θ=0.8 label clipping** is clever: if the mixed label ≥0.8, snap it to 1.0.
This prevents soft labels from diluting strong positive signal.

The **5-epoch warmup** (no mixup) lets the model learn basic features before
introducing mixed samples.

**Actionable**: Try α=1.0 + θ=0.8 + warmup in our v4b pipeline.

### 4. BCEWithLogitsLoss vs Soft CE (~+0.001-0.003)
ttahara uses standard BCE. Our soft CE creates soft targets from multi-hot vectors
and uses cross-entropy. BCE directly optimizes the binary classification threshold
for each species independently. In our v4i/v4k experiments, standard CE was worse
(−0.030 val/auroc), but that was CE (single-label), not BCE (multi-label).

**Actionable**: Try BCEWithLogitsLoss instead of soft_ce. This is a different
loss than what v4i tested (CE ≠ BCE).

### 5. Input Shape: Resized (256,256) vs Raw (224,1000) (~uncertain)
ttahara resizes mel spectrograms to a fixed (256,256) square regardless of
original time resolution. This means 5s of audio at 512 time frames gets
compressed to 256 time frames. The backbone sees a square image.

Our approach preserves the full time resolution (1000 frames for 10s at 320 hop).
More temporal detail, but the backbone processes a wider-than-tall image.

**Uncertain impact**: Square input may match ImageNet pretraining better
(ImageNet = 224×224). Our elongated input requires the model to adapt.

### 6. Variable-Length Soundscape Segments (~+0.001-0.002)
ttahara splits soundscape annotations by continuous label spans, creating
variable-length training segments (5-15s+). Our approach uses fixed 5s windows
from labeled soundscapes.

This means ttahara's model sees the complete vocalization in context, while ours
may split a call across window boundaries.

### 7. HGNetV2-B0 vs EfficientNet-B0 (~+0.001-0.003)
HGNetV2 uses heavy re-parameterization (RepVGG-style) for inference speed.
The SSLD (Simple Semi-supervised Label Distillation) pretraining may give better
features than standard ImageNet. Both are ~5-6M params.

**Actionable**: Try `hgnetv2_b0.ssld_stage2_ft_in1k` as a drop-in backbone
replacement in our pipeline. timm supports it.

### 8. Lower Weight Decay (1e-4 vs 0.01) (~+0.001-0.002)
Our 0.01 weight decay is 100× stronger. On a small dataset with aggressive
augmentation, lighter regularization may allow the model to fit better.

**Actionable**: Try weight_decay=1e-4 in our pipeline.

## Estimated Gap Attribution

| Factor | Est. Impact | Effort |
|--------|------------|--------|
| 4-fold ensemble | +0.010-0.015 | Medium (train 4×, modify inference) |
| LSEHead | +0.003-0.008 | Low (add new head option) |
| Stronger MixUp (α=1.0 + θ=0.8) | +0.002-0.005 | Low (config change) |
| BCE loss | +0.001-0.003 | Low (config change) |
| HGNetV2 backbone | +0.001-0.003 | Low (drop-in replacement) |
| Lower weight_decay | +0.001-0.002 | Low (config change) |
| Variable soundscape segments | +0.001-0.002 | Medium (data pipeline) |
| Square resize input | uncertain | Low (config change) |
| **Total estimated** | **~0.019-0.038** | |

## Recommended Next Steps (Priority Order)

### Quick wins (config changes, no new code)
1. **MixUp α=1.0, θ=0.8, warmup=5** — biggest bang for effort
2. **weight_decay=1e-4** — simple config change
3. **BCEWithLogitsLoss** — different from v4i's CE experiment
4. **HGNetV2-B0 backbone** — drop-in timm model swap

### Medium effort (new code)
5. **LSEHead** — implement as alternative to AttHead
6. **4-fold CV training** — train 4 models on PACE, ensemble at inference

### Lower priority
7. **Square resize (256×256)** — uncertain impact, may hurt temporal resolution
8. **Variable soundscape segments** — requires data pipeline changes
9. **OpenVINO inference** — faster but ONNX already works

## Reproduction Path

**Option A**: Reproduce ttahara's results on PACE
- Use our existing data pipeline, swap backbone + head + loss + augmentation
- Expected: match 0.884 or close (single-fold ~0.870, 4-fold ~0.884)

**Option B**: Apply ttahara's ideas to our EfficientNet pipeline
- Keep EfficientNet-B0, add LSEHead + BCE + stronger MixUp + 4-fold
- Expected: v4b 0.860 → ~0.875-0.885

**Option C**: Train HGNetV2-B0 with our full pipeline
- Use HGNet backbone with our SED infrastructure (PACE training, ONNX export)
- Add 4-fold ensemble at inference
- Expected: ~0.884+ (matching or exceeding ttahara)

All options can coexist with the Perch ProtoSSM pipeline — the SED model would
eventually be ensembled with Perch for a combined submission.
