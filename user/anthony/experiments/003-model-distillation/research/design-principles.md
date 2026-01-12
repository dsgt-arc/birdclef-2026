# Design Principles for Sequence Context Models

## Problem Statement

We need a lightweight sequence model that can:
1. Take static embeddings `(B, 248, 768)` as input
2. Output contextualized embeddings `(B, 248, 768)` that mimic BEATs transformer
3. Have 1-10M parameters (reasonable for distillation)

This is a **sequence-to-sequence** task modeling temporal dependencies across 248 patches.

> **Scope**: Focus on accuracy recovery. Training/eval on GPU.

## Key Principles

### 1. Depthwise Separable Convolutions

Standard convolution: `O(K × C_in × C_out × L)`
Depthwise separable: `O(K × C_in × L) + O(C_in × C_out × L)`

**Savings**: 8-9x fewer FLOPs for K=3

```
Standard:      [input] → [K×C_in×C_out conv] → [output]
Depthwise Sep: [input] → [K×1 depthwise] → [1×1 pointwise] → [output]
```

**Source**: [AM-MobileNet1D](https://arxiv.org/pdf/2004.00132)

### 2. Inverted Bottleneck

Expand→process→compress instead of compress→process→expand:

```
Traditional: 768 → 256 → 256 → 768   (bottleneck)
Inverted:    768 → 3072 → 3072 → 768 (4x expansion)
```

Depthwise convolutions are cheap, so process in expanded space.

**Source**: [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), MobileNetV2

### 3. Large Kernels

7×7 kernels outperform 3×3 with minimal FLOPs increase (+0.1 GFLOPS for +0.7% accuracy).

For 1D: use kernel size 7+ for better receptive field.

**Source**: [ConvNeXt](https://huggingface.co/learn/computer-vision-course/en/unit2/cnns/convnext)

### 4. Dilated Convolutions

Exponential receptive field growth with linear parameter increase:

```
Layer 1: dilation=1   → RF = 7
Layer 2: dilation=2   → RF = 19
Layer 3: dilation=4   → RF = 43
Layer 4: dilation=8   → RF = 91
Layer 5: dilation=16  → RF = 187
Layer 6: dilation=32  → RF = 379 (covers all 248 patches)
```

**Source**: [WaveNet](https://arxiv.org/pdf/1609.03499), [TCN](https://arxiv.org/pdf/1803.01271)

### 5. LayerNorm + Residuals

- LayerNorm instead of BatchNorm (ConvNeXt finding)
- Skip connections for gradient flow

## Architecture Comparison

| Architecture | Type | Complexity | Params | Library |
|--------------|------|------------|--------|---------|
| **Mamba2** | SSM | O(n) | ~4M | transformers |
| **ConvNeXt1D** | CNN | O(n) | ~2.5M | custom |
| **TCN** | CNN | O(n) | ~2-4M | pytorch-tcn |
| **MLPMixer** | MLP | O(n) | ~4M | custom |
| **TinyTransformer** | Attention | O(n²) | ~2M | custom |

## Recommendation

**Primary candidates**:
1. **Mamba2** - State-of-the-art SSM, O(n), pure PyTorch via HuggingFace
2. **ConvNeXt1D** - Modern CNN with proven design, efficient

**Baselines**:
3. **TCN** - Well-tested, pip installable
4. **TinyTransformer** - Shows attention baseline (O(n²) but 248 tokens is fine)

## References

- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
- [MobileNetV3](https://arxiv.org/abs/1905.02244)
- [TCN](https://arxiv.org/pdf/1803.01271)
- [WaveNet](https://arxiv.org/pdf/1609.03499)
- [EfficientAT](https://github.com/fschmid56/EfficientAT)
