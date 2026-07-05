# Concept: SSAMBA (Self-Supervised Audio Mamba)

**Status**: current research thread
**Created**: 2026-01-15
**Updated**: 2026-02-17

## Idea

SSAMBA is the first self-supervised, attention-free audio representation model based on bidirectional Mamba state space models. It replaces the quadratic-complexity transformer in SSAST with linear-complexity Mamba blocks, achieving **92.7% faster inference** and **95.4% better memory efficiency** while maintaining competitive accuracy.

This is the backbone architecture for our BirdCLEF pipeline, validated through experiments 004-009.

## Architecture

### Core Structure

```
Mel spectrogram (1, 128, T)
         ↓
AudioPatchEmbed (Conv2d) → patch tokens (B, N, D)
         ↓
+ learned positional embeddings (1, N+1, D)
+ CLS token (inserted at middle of sequence)
         ↓
24 bidirectional Mamba layers (paired: even=forward, odd=flipped)
         ↓
LayerNorm → token embeddings (B, N+1, D)
         ↓
Task head (classification, SED, or reconstruction)
```

### Two Levels of Bidirectionality

1. **Inner (per-block)**: each `BidirectionalMamba` module has separate forward and backward SSMs (bimamba_type="v2") with independent A, B, C, D, conv1d parameters
2. **Outer (layer pairs)**: layers processed in pairs — even layers see normal input, odd layers see `.flip([1])`, results summed

**Consequence**: every token's output embedding reflects the **entire sequence** from both directions. There is no single "state compression" token — each position has a different perspective on the same global information.

### Model Sizes

| Model | Params | ESC-50 | AudioSet-20K mAP | embed_dim | d_state |
|-------|--------|--------|------------------|-----------|---------|
| SSAMBA-tiny | 7M | 80.1% | 23.3 | 192 | 16 |
| SSAMBA-small | 26M | 85.5% | 25.6 | 384 | 16 |
| SSAMBA-base | 99M | 89.3% | 28.3 | 768 | 16 |

### Key Implementation Details

- **CLS token**: inserted at `num_patches // 2` (middle of sequence), not at the beginning
- **Positional embeddings**: learned, shape `(1, num_patches + 1, embed_dim)`, tied to input length
- **Double-norm quirk**: `norm_f` is applied twice in the original SSAMBA code (a bug replicated for checkpoint compatibility)
- **Patch sizes**: originally 16x16; experiment 005 added support for rectangular patches (128x1 for 10ms/token SED)
- **CUDA kernels**: required via `mamba-ssm` for practical training; pure PyTorch SSM has O(L) Python loop

## Experimental Results

| Experiment | Configuration | Result | Key Finding |
|------------|--------------|--------|-------------|
| 004 | HuggingFace port | Working | SSAMBA ported with full weight loading |
| 005 | Rectangular patches (128x1) | Working | 10ms/token resolution, ONNX benchmark |
| 006 | BirdSet HSN from scratch | **0.086 cmAP** | Far below baselines — needs pretraining |
| 007 | AudioSet pretrained transfer | **0.173 cmAP** | Pretraining 2x lift; Mamba-only transfer ≈ full |
| 008 | Attention-pooled SED | **0.164 cmAP** | SED attention didn't help — bottleneck is elsewhere |
| 009 | SSL pretrain + finetune pipeline | In progress | L1 + linear decoder, Kaggle 0.484 |

### Key Findings

1. **Pretraining is the dominant factor**: 0.086 (scratch) → 0.173 (pretrained), 2x lift
2. **Mamba-only transfer works**: transferring only the Mamba layers (96.9% of params, skipping patch_embed + pos_embed) matches full transfer
3. **Severe overfitting**: train cmAP >0.95, val ~0.17 — the 7M param model memorizes but doesn't generalize
4. **Capacity gap**: SSAMBA Tiny 0.17 vs ConvNeXt 0.48 vs SOTA 0.55 — likely model size, not architecture
5. **SED attention pooling didn't help**: bottleneck is representation quality, not pooling strategy
6. **Scaling model size didn't help**: SSAMBA Small (25.5M) achieved same 0.165 cmAP as Tiny (7M)

## Open Questions

- **Can larger SSAMBA close the gap?** Small (26M) didn't improve over Tiny. Is the issue architecture-fundamental, or does it need even more scale?
- **Does bidirectional processing prevent state compression?** Reference-conditioned use remains an open research question.
- **Would a CNN frontend help?** This remains an open architecture question, but the separate speculative frontend note was removed during cleanup.
- **Is the double-norm bug hurting fine-tuning?** Could test removing it after transfer.

## Connections

- **HF port**: `user/anthony/experiments/004-ssamba-hf-port/ssamba_hf/`
- **Related concepts**: [Temporal resolution and sequence length](temporal-resolution-sequence-length.md), [Working note framing](working-note.md)
- **Vendor docs**: [SSAMBA](../vendor/ssamba.md), [SSAST](../vendor/ssast.md)
- **References**: [SSAMBA Paper (SLT 2024)](https://arxiv.org/abs/2405.11831), [Mamba Paper](https://arxiv.org/abs/2312.00752)
