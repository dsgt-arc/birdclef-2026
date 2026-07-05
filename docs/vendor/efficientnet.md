---
name: "EfficientNet"
source: "https://github.com/huggingface/pytorch-image-models"
license: "Apache-2.0"
category: "pretrained-model"
pipeline_stage:
  - "embedding"
  - "inference"
relevance: "CNN backbone for SED models in BirdCLEF pipeline"
status: "active-reference"
---

# EfficientNet

## Overview

EfficientNet is a family of CNN architectures designed via neural architecture search (NAS)
that uniformly scales depth, width, and resolution using compound scaling coefficients.
Originally published by Tan & Le (2019), it achieved state-of-the-art ImageNet accuracy
at the time with significantly fewer parameters than alternatives. We use it via the
`timm` library as a feature extractor backbone for sound event detection (SED).

## Architecture

EfficientNet-B0 (the base model) uses Mobile Inverted Bottleneck (MBConv) blocks with
squeeze-and-excitation (SE) attention. The architecture was found via NAS on the
MnasNet search space, then scaled up to B1-B7 using compound scaling.

**Key design choices:**
- **MBConv blocks**: depthwise separable convolutions with inverted residuals
- **Squeeze-and-Excitation (SE)**: channel attention after depthwise conv
- **Swish activation**: smooth, non-monotonic (x * sigmoid(x))
- **Compound scaling**: balances depth/width/resolution uniformly

**EfficientNet-B0 feature map stages:**

| Stage | Output Resolution | Channels |
|-------|------------------|----------|
| 0 | 112x112 | 16 |
| 1 | 56x56 | 24 |
| 2 | 28x28 | 40 |
| 3 | 14x14 | 112 |
| 4 | 7x7 | **320** |

Final embed_dim = **320** (last stage channels in `features_only` mode).
After conv head expansion: **1280** (used for classification, not in our SED pipeline).

## EfficientNet Variants in timm

### V1 (used in our pipeline)

| Model | Params | GMACs | Top-1 | embed_dim | Notes |
|-------|--------|-------|-------|-----------|-------|
| `efficientnet_b0` | 5.3M | 0.4 | ~77.7% | 320 | Default timm weights |
| `efficientnet_b0.ra_in1k` | 5.3M | 0.4 | ~77.7% | 320 | RandAugment recipe (ResNet Strikes Back) |
| `tf_efficientnet_b0.ns_jft_in1k` | 5.3M | 0.4 | 78.7% | 320 | Noisy Student, JFT pretrained |

### V2 (not yet tested)

EfficientNetV2 (Tan & Le, 2021) replaces MBConv with Fused-MBConv in early stages
(regular conv instead of depthwise separable) for better GPU utilization, and uses
progressive learning (increasing image size + augmentation during training).

| Model | Params | GMACs | Top-1 | embed_dim* | Notes |
|-------|--------|-------|-------|-----------|-------|
| `tf_efficientnetv2_b0.in1k` | 7.1M | 0.5 | 78.4% | 192 | V2 base, ImageNet-1k only |
| `efficientnetv2_rw_t.ra2_in1k` | 13.7M | 1.9 | 82.4% | 208 | timm redesign, RA2 recipe, 1024 after conv head |
| `tf_efficientnetv2_s.in21k_ft_in1k` | 21.5M | 5.4 | 83.9% | 256 | ImageNet-21k pretrained, largest/strongest |

*embed_dim = last-stage feature map channels in `features_only` mode (what our SED head receives).

**Key V2 differences from V1:**
- Fused-MBConv in early stages (better GPU throughput, fewer memory-bound ops)
- Progressive learning during training
- Slightly larger parameter counts at equivalent "B0" tier
- V2-S with 21k pretraining is significantly stronger but 4x the compute

## Our Usage

We use `efficientnet_b0` (V1) as the SED backbone in `features_only=True` mode with
`in_chans=1` (mono mel spectrogram input). The last-stage 320-dim features feed into
the SED attention/LSE head.

**Config** (`src/birdclef/encoders/efficientnet.py`):
```python
timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    features_only=True,
    in_chans=1,           # mono mel spectrogram
    drop_path_rate=0.0,
)
# embed_dim = 320 (from backbone.feature_info.channels()[-1])
```

**Experiment results** (exp 019, single-fold):
- v4b (EfficientNet-B0, old recipe): 0.865 val / **0.860 LB**
- v5f (EfficientNet-B0, ttahara recipe): 0.866 val / **0.855 LB**

The ttahara recipe barely helped EfficientNet (+0.001 val, -0.005 LB), suggesting
the B0 backbone is the bottleneck, not the training recipe.

## Potential Upgrades

| Candidate | Why interesting | Risk |
|-----------|----------------|------|
| `efficientnetv2_rw_t.ra2_in1k` | 82.4% top-1, 208-dim features, 13.7M params, 224px input | 2.6x params, needs ONNX export validation |
| `tf_efficientnetv2_s.in21k_ft_in1k` | 83.9% top-1, 21k pretrained, 256-dim features | 21.5M params, 5.4 GMACs, may be too heavy for Kaggle kernel |
| `tf_efficientnet_b0.ns_jft_in1k` | Same size, +1% top-1 from Noisy Student pretraining | Minimal change, may not move the needle |

## References

- [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946) (Tan & Le, 2019)
- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) (Tan & Le, 2021)
- [timm EfficientNet docs](https://huggingface.co/docs/timm/models/efficientnet)
