# Concept: SSAST (Self-Supervised Audio Spectrogram Transformer)

**Status**: historical
**Created**: 2026-01-15
**Updated**: 2026-01-15

## Idea

SSAST is a self-supervised pretraining framework for Audio Spectrogram Transformers (AST) that uses joint discriminative and generative masked spectrogram patch modeling (MSPM). Unlike supervised ImageNet pretraining, SSAST learns audio-specific representations from unlabeled audio data (AudioSet + LibriSpeech), supporting arbitrary patch shapes and sizes.

The key innovation is treating audio spectrograms as 2D images and applying Vision Transformer (ViT) style patching with self-supervised objectives similar to BERT/MAE.

## Motivation

- **Baseline for efficient audio models**: SSAST represents the transformer-based approach to audio SSL. Understanding its architecture and performance is essential for comparing against more efficient alternatives (SSAMBA, distilled models).
- **Pretraining without labels**: Can leverage the massive unlabeled AudioSet corpus for pretraining, then finetune on small labeled datasets like ESC-50 or BirdCLEF.
- **Flexibility**: Supports both patch-based (for audio events) and frame-based (for speech) configurations.

## Background

### Architecture

- **Input**: Audio waveform converted to 128-dim log Mel filterbank features
- **Patching**: 16x16 patches with 6-overlap in time and frequency during finetuning
- **Embedding**: Linear projection to 768-dim patch embeddings + learnable positional embeddings
- **Backbone**: Standard Transformer encoder (12 layers for base model)
- **Pretraining**: Masked spectrogram patch modeling (MSPM) - predict masked patches using both discriminative (contrastive) and generative (reconstruction) objectives

### Published Results

| Model | ESC-50 | AudioSet | Params |
|-------|--------|----------|--------|
| AST (supervised) | 95.6% | - | 87M |
| SSAST-base | ~88% | ~31 mAP | 87M |
| SSAST-tiny | ~80% | ~23 mAP | ~7M |

### References

- [SSAST Paper (AAAI 2022)](https://arxiv.org/abs/2110.09784)
- [Official Code](https://github.com/YuanGongND/ssast)
- [AST Paper (Interspeech 2021)](https://arxiv.org/abs/2104.01778)

### Vendor Documentation

- [SSAST Vendor Doc](../vendor/ssast.md) - Local code at `vendor/ssast/`
- [AST Vendor Doc](../vendor/ast.md) - upstream/HuggingFace AST reference

## Open Questions

- How does SSAST compare to BEATs on bird-specific audio classification?
- Can we use SSAST as a teacher model instead of/alongside BEATs?
- What is the actual inference time on CPU for SSAST-tiny vs BEATs?

## Testable Hypotheses

1. SSAST-tiny achieves comparable ESC-50 accuracy to BEATs linear probe (~84%) while being faster
2. SSAST embeddings are more amenable to distillation than BEATs due to simpler architecture
3. Combining SSAST and BEATs as ensemble teachers improves student distillation

## Connections

- **Based on**: Vision Transformer (ViT), BERT masked prediction, Audio Spectrogram Transformer (AST)
- **Related to**: [ssamba-audio-mamba](ssamba-audio-mamba.md), [mlp-mixer-distillation](mlp-mixer-distillation.md)
- **Vendor docs**: [SSAST](../vendor/ssast.md), [AST](../vendor/ast.md), [BEATs](../vendor/beats.md)
- **Experiment**: [001-encoder-benchmark](../../user/anthony/experiments/001-encoder-benchmark/proposal.md) Part C
- **Could lead to**: Using SSAST-tiny as a faster teacher, or as a student architecture baseline

## Notes

- SSAST is the transformer baseline against which SSAMBA (Mamba-based) shows 92.7% speedup
- The quadratic attention complexity makes SSAST slow for long audio sequences
- For BirdCLEF inference budget, SSAST-base is likely too slow; SSAST-tiny might work
- Pretraining requires ~10 days on 4x Titan X GPUs
