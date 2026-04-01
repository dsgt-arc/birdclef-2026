---
name: "PP-HGNetV2"
source: "https://github.com/PaddlePaddle/PaddleClas"
license: "Apache-2.0"
category: "pretrained-model"
pipeline_stage:
  - "embedding"
  - "inference"
relevance: "CNN backbone for SED models in BirdCLEF — outperforms EfficientNet on LB"
status: "active"
---

# PP-HGNetV2

## Overview

PP-HGNetV2 (High-Performance GPU Network V2) is a CNN architecture from Baidu's PaddlePaddle
team, designed to maximize GPU throughput while maintaining strong accuracy. Unlike
EfficientNet which optimizes for FLOP efficiency (often at the cost of actual GPU speed),
HGNetV2 prioritizes operations that run fast on modern GPUs: large regular convolutions,
minimal branching, and hardware-friendly channel dimensions.

Available in timm as `hgnetv2_b0.ssld_stage2_ft_in1k` and other variants.

## Architecture

### Design Philosophy: GPU-Friendly Over FLOP-Efficient

The key insight behind HGNet is that **FLOPs don't equal latency on GPUs**. Operations
like depthwise separable convolutions (used heavily in EfficientNet/MobileNet) have low
FLOPs but poor GPU utilization due to memory bandwidth bottlenecks. HGNet instead uses
operations with high arithmetic intensity that saturate GPU compute units.

**Design principles:**
1. **Prefer regular convolutions** over depthwise separable (higher GPU utilization)
2. **Use LearnableAffineBlock (LR-Pad)** — lightweight channel mixing after convs
3. **Minimize branching** — sequential blocks, not multi-branch like Inception
4. **GPU-friendly channel dims** — multiples of 8/16 for tensor core alignment
5. **Stage-wise design** — progressive downsampling with increasing channels

### HG Block (Core Building Block)

The HG (High-performance GPU) block is the fundamental unit. Unlike MBConv (inverted
bottleneck with depthwise conv), HG blocks use standard convolutions in a dense
connectivity pattern inspired by HarDNet:

- Multiple 3x3 regular convolutions with dense connections within the block
- LearnableAffineBlock (LR-Pad) after each conv for lightweight feature refinement
- No squeeze-and-excitation attention (saves memory bandwidth)
- Aggregation layer to combine multi-scale features within the block

### HGNetV2-B0 Architecture

| Stage | Output Resolution | Channels | Notes |
|-------|------------------|----------|-------|
| Stem | 112x112 | 16 | Conv + pooling |
| 1 | 56x56 | 32 | |
| 2 | 28x28 | 64 | |
| 3 | 14x14 | 128 | |
| 4 | 7x7 | **1024** | Large channel expansion |

Final embed_dim = **1024** (last stage channels in `features_only` mode).

### SSLD Training (Knowledge Distillation)

The `ssld_stage2_ft_in1k` weights use Baidu's SSLD (Simple Semi-supervised Label
Distillation) — a two-stage knowledge distillation process:
1. **Stage 1**: Train a large teacher model
2. **Stage 2**: Distill into the smaller model using soft labels + unlabeled data

This training recipe gives HGNetV2-B0 a significant accuracy boost over standard
ImageNet training (similar in spirit to Google's Noisy Student training).

## Model Specs

| Property | HGNetV2-B0 |
|----------|-----------|
| Parameters | ~6M |
| embed_dim (features_only) | **1024** |
| ImageNet Top-1 | ~78-79% (with SSLD) |
| Input Resolution | 224x224 |
| Activation | ReLU (GPU-optimized) |
| Attention | None (no SE blocks) |

## Our Usage

We use `hgnetv2_b0.ssld_stage2_ft_in1k` as the SED backbone via timm's generic
encoder support (`features_only=True`, `in_chans=1`).

**Config:**
```python
timm.create_model(
    "hgnetv2_b0.ssld_stage2_ft_in1k",
    pretrained=True,
    features_only=True,
    in_chans=1,
)
# embed_dim = 1024
```

**Experiment results** (exp 019, single-fold):
- v5e (HGNet, old recipe): 0.861 val — backbone alone didn't help
- v5g (HGNet, ttahara recipe): 0.872 val / **0.892 LB** — breakthrough
- v5j (HGNet, AmplitudeToDB): 0.874 val / **0.906 LB** — best single-fold
- v5j-4fold: **0.913 LB** — best trained model

## HGNetV2 Variant Family

| Variant | Params | Top-1 | Use case |
|---------|--------|-------|----------|
| B0 | ~6M | ~78-79% | Lightweight, our current choice |
| B1 | ~8M | ~79-80% | Slightly larger |
| B2 | ~11M | ~81% | Mid-range |
| B3 | ~16M | ~82% | |
| B4 | ~25M | ~83% | Heavy |
| B5 | ~40M | ~84% | Very heavy |
| B6 | ~75M | ~86% | Classification-focused |

## References

- [PP-HGNetV2 docs (PaddleClas)](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/models/ImageNet1k/PP-HGNetV2.md)
- [PP-HGNet: High-Performance GPU Network](https://arxiv.org/abs/2309.02359)
- [timm model card](https://huggingface.co/timm/hgnetv2_b0.ssld_stage2_ft_in1k)
- [ttahara's HGNetV2-B0 BirdCLEF notebook](../../user/murilo/notes/ttahara-hgnetv2-b0-annotated.md)
