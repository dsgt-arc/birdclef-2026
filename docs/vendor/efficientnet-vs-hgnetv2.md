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

## Why HGNet Wins: Three Hypotheses

### 1. embed_dim: 1024 vs 320 (most likely factor)

This is probably the single biggest reason. The SED head receives the last-stage
feature map and must produce per-frame, per-species predictions for 234 classes.

- **EfficientNet-B0 gives the head 320 channels** to work with
- **HGNetV2-B0 gives the head 1024 channels** — 3.2x more capacity

With 234 bird species (many rare, many acoustically similar), 320 dimensions is
a tight bottleneck. The head must compress 320 → 234 predictions, leaving almost
no room for learned intermediate representations. With 1024 dimensions, the head
has much more information to work with.

This is consistent with our results:
- v5e (HGNet + old recipe): 0.861 val — same as v4b (EfficientNet + old recipe)
- The raw backbone swap was neutral on *offline val* because the val set is small
  and forgiving, but the extra capacity matters on the harder LB distribution

### 2. GPU-Friendly Ops = Faster Training = More Effective Epochs

HGNetV2 trains ~8-10x faster than EfficientNet under the ttahara recipe:
- v5g (HGNet): **21 min** for 20 epochs
- v5f (EfficientNet): similar wall time but EfficientNet's depthwise ops
  underutilize the GPU

EfficientNet's MBConv blocks use depthwise separable convolutions — low FLOPs
but memory-bandwidth-bound on GPUs. HGNet's regular convolutions have higher
arithmetic intensity and better saturate GPU compute units. This means:
- Each training step is more GPU-efficient
- The optimizer gets more useful gradient signal per unit of wall time
- OneCycleLR's 20-epoch schedule is better matched to HGNet's convergence

### 3. Regular Convolutions May Suit Spectrograms Better

Mel spectrograms have different spatial structure than natural images:
- **Frequency axis**: semantically meaningful (pitch), not translationally invariant
- **Time axis**: sequential, with variable-duration events
- **Textures**: harmonic stacks, broadband noise, frequency-modulated sweeps

Depthwise separable convolutions (EfficientNet) process each channel independently
before mixing — this makes sense for RGB images where channels are correlated but
distinct. For single-channel mel spectrograms, **regular convolutions** (HGNet) that
jointly process spatial neighborhoods may capture spectro-temporal patterns more
naturally.

Additionally, HGNet's dense connections within each HG block provide multi-scale
feature aggregation at each stage — useful for detecting bird calls that span
different frequency/time scales simultaneously.

## Experimental Evidence

### Same Recipe, Different Backbone (isolating architecture effect)

| Model | Backbone | embed_dim | val/auroc | LB | LB delta |
|-------|----------|----------|----------|-----|----------|
| v5f | EfficientNet-B0 | 320 | 0.866 | 0.855 | baseline |
| v5g | HGNetV2-B0 | 1024 | 0.872 | **0.892** | **+0.037** |

Both use identical ttahara recipe (BCE, LSEHead, MixUp, OneCycleLR, 20 epochs).
The **only** difference is the backbone. The +0.037 LB gap is entirely from
the architecture change.

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
| EfficientNet-B0 (current) | 5.3M | 77.7% | 320 | Baseline, bottlenecked |
| EfficientNetV2-B0 | 7.1M | 78.4% | 192 | *Worse* embed_dim, unlikely to help |
| EfficientNetV2-RW-T | 13.7M | 82.4% | 208 | Better accuracy but still low embed_dim |
| EfficientNetV2-S (21k) | 21.5M | 83.9% | 256 | Strongest, but 4x compute and embed_dim still < HGNet |
| **HGNetV2-B0** (current) | **~6M** | **78-79%** | **1024** | **Winner** |

*embed_dim = last-stage channels in features_only mode (what our SED head sees).

**Critical insight**: All EfficientNetV2 variants have embed_dim of 192-256 in
features_only mode — *worse* than V1's 320, and far below HGNet's 1024. The V2
models expand to 1280 via a conv head, but in `features_only=True` mode (which
we use for SED), that expansion doesn't apply.

If the embed_dim hypothesis is correct, switching to EfficientNetV2 would likely
**regress** performance unless we also modify the encoder to include the conv head
expansion layer. Even then, 1280 would be only marginally above HGNet's 1024.

## Recommendations

### Keep HGNetV2-B0 as Default Backbone

HGNet is the clear winner for our SED pipeline:
- 3.2x more feature dimensions for the SED head (1024 vs 320)
- Faster training (~21 min vs hours)
- Better LB scores across all recipe variants
- Well within Kaggle kernel budget (6M params, fast ONNX inference)

### If Exploring Larger Backbones

| Candidate | Params | embed_dim | Why |
|-----------|--------|-----------|-----|
| HGNetV2-B1 | ~8M | higher | Natural next step, same architecture family |
| HGNetV2-B2 | ~11M | higher | More capacity, still reasonable for Kaggle |
| EfficientNetV2-RW-T | 13.7M | 208 (1024 w/head) | Only if we include conv head expansion |

### Don't Bother With

- **EfficientNetV2-B0**: Lower embed_dim (192) than V1 (320), no advantage
- **EfficientNet-B0 variants** (NS-JFT, RA recipes): Same 320 embed_dim bottleneck
- **EfficientNetV2-S**: 21.5M params is heavy for Kaggle kernel, and embed_dim
  (256) is still below HGNet

## Summary

HGNetV2-B0 outperforms EfficientNet-B0 primarily because it provides **3.2x more
feature dimensions** (1024 vs 320) to the SED head, while being a similar-sized
model (~6M vs 5.3M params). Its GPU-friendly architecture (regular convolutions,
no SE attention, ReLU) also enables faster training. The ttahara training recipe
was designed for this architecture class and synergizes with HGNet but not with
EfficientNet. EfficientNetV2 variants don't solve the embed_dim bottleneck in
`features_only` mode and are unlikely to close the gap.
