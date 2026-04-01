---
name: "EfficientNet vs HGNetV2 Comparison"
source: "n/a"
license: "n/a"
category: "pretrained-model"
pipeline_stage:
  - "embedding"
  - "inference"
relevance: "Backbone architecture comparison for BirdCLEF SED pipeline"
status: "reference"
---

# EfficientNet vs HGNetV2: Why HGNet Wins for SED

## The Question

In exp 019, switching from EfficientNet-B0 to HGNetV2-B0 (same training recipe)
improved LB from 0.855 to 0.892 (+0.037). Both are ~5-6M parameter CNNs with
similar ImageNet accuracy (~78%). Why does HGNet outperform so dramatically?

## Architecture Comparison

| Property | EfficientNet-B0 | HGNetV2-B0 |
|----------|----------------|-----------|
| **Parameters** | 5.3M | ~6M |
| **ImageNet Top-1** | ~77.7% | ~78-79% (SSLD) |
| **embed_dim** (features_only) | **320** | **1024** |
| **Core block** | MBConv (depthwise separable) | HG Block (regular conv, dense connections) |
| **Attention** | Squeeze-and-Excitation (SE) | None |
| **Activation** | Swish (x * sigmoid(x)) | ReLU |
| **Design goal** | FLOP efficiency | GPU throughput |
| **Scaling method** | NAS + compound scaling | Manual stage design |
| **Pretraining** | Standard ImageNet | SSLD (knowledge distillation) |

## Why HGNet Wins: Experimentally Verified

We ran a 2×2 ablation grid (exp 019, v5f/v5m/v5n/v5j) to isolate the three
candidate factors. The results are conclusive.

### 1. embed_dim: 1024 vs 320 — CONFIRMED as primary factor (45% of LB gap)

The SED head receives the last-stage feature map and must produce per-frame,
per-species predictions for 234 classes.

- **EfficientNet-B0 gives the head 320 channels** to work with
- **HGNetV2-B0 gives the head 1024 channels** — 3.2x more capacity

We tested this directly by adding a 1x1 Conv2d projection (320→1024) after
EfficientNet's backbone (v5m). This single layer — 328K parameters — improved
LB by **+0.023** (v5n 0.875 → v5m 0.898), the largest single factor.

With 234 bird species (many rare, many acoustically similar), 320 dimensions is
a tight bottleneck. The head must compress 320 → 234 predictions, leaving almost
no room for learned intermediate representations. With 1024 dimensions, the head
has much more information to work with.

**Offline val completely missed this.** The projection added only +0.002 val/auroc
but +0.023 LB, a ~10x amplification. The offline val set is too small and
forgiving to detect the capacity bottleneck that matters on the harder test
distribution.

### 2. Mel pipeline — secondary factor (39% of LB gap)

Switching from ttahara mel (torchaudio, resize 256×256, min-max norm) to v5j mel
(NNAudio, AmplitudeToDB norm, no resize, native 256×511) contributed +0.020 LB
(v5f 0.855 → v5n 0.875). This is a meaningful gain from the preprocessing change,
independent of the backbone.

### 3. HGNet native architecture — minor factor (16% of LB gap)

After matching embed_dim via projection, HGNet still has a small edge:
v5m (0.898) → v5j (0.906) = +0.008 LB. This residual comes from HGNet's
architectural properties:
- **Regular convolutions** vs depthwise separable (better GPU utilization,
  potentially better for single-channel spectrograms)
- **Dense connections** within HG blocks (multi-scale feature aggregation)
- **SSLD pretraining** (knowledge distillation, qualitatively different features)
- **ReLU** vs Swish activation (faster, may regularize differently)

These matter, but are secondary to the embed_dim bottleneck.

## Experimental Evidence

### 2×2 Ablation Grid (exp 019, v5f/v5n/v5m/v5j)

Isolates mel pipeline vs embed_dim (projection) on LB:

|  | ttahara mel | v5j mel (NNAudio+AmpToDB) | Delta (mel) |
|--|------------|--------------------------|-------------|
| **EffNet 320 (no proj)** | v5f: 0.855 LB | v5n: 0.875 LB | **+0.020** |
| **EffNet 320→1024 (proj)** | — | v5m: 0.898 LB | — |
| **Delta (projection)** | — | **+0.023** | — |
| **HGNet 1024 (native)** | v5g: 0.892 LB | v5j: 0.906 LB | +0.014 |

**Factor decomposition** (v5f 0.855 → v5j 0.906, total gap = 0.051):

| Factor | LB gain | Share | How measured |
|--------|---------|-------|-------------|
| Mel pipeline | +0.020 | 39% | v5f → v5n (same backbone, different mel) |
| **Projection (embed_dim)** | **+0.023** | **45%** | v5n → v5m (same mel, add 1x1 Conv2d 320→1024) |
| HGNet native architecture | +0.008 | 16% | v5m → v5j (proj EffNet → native HGNet) |

### Offline vs LB Disconnect

| Factor | Offline delta | LB delta | Amplification |
|--------|-------------|----------|---------------|
| Mel pipeline (v5f→v5n) | +0.005 val | +0.020 LB | 4x |
| Projection (v5n→v5m) | +0.002 val | +0.023 LB | **~12x** |
| HGNet native (v5m→v5j) | +0.001 val | +0.008 LB | 8x |

The projection's +0.023 LB from +0.002 offline is the most extreme
val→LB amplification. Offline metrics cannot detect this bottleneck.

### Same Backbone, Different Recipe (isolating recipe effect)

| Model | Backbone | Recipe | val/auroc | LB |
|-------|----------|--------|----------|-----|
| v4b | EfficientNet-B0 | Old (soft CE, AttHead, 50ep) | 0.865 | 0.860 |
| v5f | EfficientNet-B0 | ttahara | 0.866 | 0.855 |
| v5e | HGNetV2-B0 | Old | 0.861 | — |
| v5g | HGNetV2-B0 | ttahara | 0.872 | **0.892** |

The ttahara recipe **barely helps EfficientNet** (0.860 → 0.855 LB, regressed!)
but **unlocks HGNet** (0.861 → 0.872 val, 0.892 LB). The recipe and backbone
are synergistic — the recipe was designed *for* HGNet-class architectures.

### 4-Fold Ensemble Comparison

| Ensemble | LB |
|----------|-----|
| v5g-4fold (HGNet) | **0.901** |
| v5j-4fold (HGNet, AmplitudeToDB) | **0.913** |
| EfficientNet ensembles | Not attempted (single-fold already behind) |

## What About EfficientNetV2?

EfficientNetV2 addresses some of EfficientNet's GPU inefficiency by replacing
MBConv with Fused-MBConv (regular conv) in early stages. This makes it
architecturally closer to HGNet in the early layers.

| Model | Params | Top-1 | embed_dim* | Potential |
|-------|--------|-------|-----------|-----------|
| EfficientNet-B0 (current) | 5.3M | 77.7% | 320 | Bottlenecked — confirmed |
| EfficientNetV2-B0 | 7.1M | 78.4% | 192 | *Worse* embed_dim, would need projection |
| EfficientNetV2-RW-T | 13.7M | 82.4% | 208 | Better features but still needs projection |
| EfficientNetV2-S (21k) | 21.5M | 83.9% | 256 | Strongest features, 4x compute, needs projection |
| **HGNetV2-B0** (current) | **~6M** | **78-79%** | **1024** | **Winner — native high embed_dim** |

*embed_dim = last-stage channels in features_only mode (what our SED head sees).

All EfficientNetV2 variants have embed_dim of 192-256 in `features_only` mode —
*worse* than V1's 320. However, now that we've confirmed the projection approach
works (v5m: +0.023 LB from 320→1024), any backbone + projection is viable.
EfficientNetV2-RW-T (82.4% top-1) + projection could theoretically outperform
HGNet by providing stronger base features to project from.

## Recommendations

### Keep HGNetV2-B0 as Default Backbone

HGNet remains the best option — it provides high embed_dim natively without
requiring a projection layer:
- Native 1024-dim features (no projection overhead)
- +0.008 LB over projected EfficientNet (small but free)
- Faster training (~21 min for 20 epochs)
- Well within Kaggle kernel budget (6M params, fast ONNX inference)

### If Exploring Alternatives

Now that embed_dim is confirmed as the bottleneck, any backbone + projection
is a viable strategy. The question becomes: which backbone produces the best
320-dim features to project from?

| Candidate | Params | Native embed_dim | Strategy | Why |
|-----------|--------|-----------------|----------|-----|
| HGNetV2-B1/B2 | ~8-11M | higher | Native | More capacity, same arch family |
| EfficientNetV2-RW-T + proj | 13.7M | 208→1024 | Projection | 82.4% top-1, stronger features |
| EfficientNet-B0 + proj | 5.3M | 320→1024 | Projection | Already proven at 0.898 LB |

### Key Insight for Future Backbones

**Always check embed_dim in `features_only` mode.** Any backbone with
embed_dim < ~512 will likely need a projection layer for 234-class SED.
The cost is minimal (~328K params for a 1x1 Conv2d) but the LB impact
is massive (+0.023 in our case).

## Summary

HGNetV2-B0 outperforms EfficientNet-B0 primarily because of **embed_dim**
(1024 vs 320). This was confirmed experimentally with a 2×2 ablation grid:

- **embed_dim (projection 320→1024): +0.023 LB (45% of gap)** — verified
  by adding a single 1x1 Conv2d layer to EfficientNet (v5n→v5m)
- Mel pipeline (NNAudio+AmplitudeToDB): +0.020 LB (39%) — v5f→v5n
- HGNet native architecture: +0.008 LB (16%) — v5m→v5j

The offline val metric completely missed the embed_dim bottleneck (+0.002
val vs +0.023 LB, ~12x amplification). Any future backbone evaluation
should prioritize embed_dim and validate on LB, not offline val.

Reference: `user/murilo/experiments/019-sed-ablations/results.md` for
full 2×2 grid analysis and detailed per-variant metrics.
