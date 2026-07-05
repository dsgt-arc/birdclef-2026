# Proposal: Foundation Model Encoder Benchmark

**Status**: in-progress
**Created**: 2025-01-09
**Updated**: 2026-01-15
**Author**: anthony

## Related Documentation

- **Concepts**:
  - [LoRA Finetuning for Audio Transformers](../../../docs/concepts/lora-finetuning-audio.md) - Parameter-efficient finetuning approach
  - [SSAST Audio Transformer](../../../docs/concepts/ssast-audio-transformer.md) - Self-supervised AST baseline
  - [SSAMBA Audio Mamba](../../../docs/concepts/ssamba-audio-mamba.md) - Efficient Mamba-based alternative
- **References**:
  - [SSL Pretraining Objectives](../../../docs/references/ssl-pretraining-objectives.md) - Why prediction-based SSL (BEATs) outperforms reconstruction-based (EAT) for linear probes
  - [BEATs Architecture](../../../docs/references/deep-research/beats-architecture.md) - Deep dive into BEATs model
  - [Wav2Vec2 Landscape](../../../docs/references/deep-research/wav2vec2-landscape.md) - Speech foundation model context

## Hypothesis

BEATs and EAT will significantly outperform wav2vec2 on audio classification because they were pretrained on AudioSet, while wav2vec2 was pretrained on speech (LibriSpeech). We expect to match published results on ESC-50 (sanity check) and see similar patterns on AudioSet bird classes.

**Extended Hypothesis (Part C)**: SSAMBA will achieve comparable accuracy to transformer-based models (AST, SSAST, BEATs) while being significantly faster due to linear complexity. This informs whether we can use SSAMBA directly for BirdCLEF or need distillation.

## Background

Before exploring distillation strategies, we need baseline performance numbers for candidate encoders. Using AudioSet's bird classes eliminates domain shift as a confounding variable - we're measuring pure encoder quality on the exact distribution they were trained on.

This informs our choice of teacher model for distillation.

## Method

### Approach

Evaluate frozen feature extraction from pretrained encoders on ESC-50 benchmark:

**Phase 1 (Complete)**: Core encoders
- wav2vec2, BEATs, EAT

**Phase 2 (New)**: Transformer vs Mamba comparison
- AST (supervised ImageNet → AudioSet)
- SSAST (self-supervised AudioSet + LibriSpeech)
- SSAMBA (Mamba-based, self-supervised)

Use linear probe to isolate encoder quality. Measure both accuracy and inference time to inform architecture selection for BirdCLEF.

### Setup

| Component | Details |
|-----------|---------|
| Data | ESC-50 (2k clips, 50 classes) |
| Compute | Single GPU, ~1-2 hours per model |
| Dependencies | `transformers`, BEATs (`vendor/unilm/beats`), EAT (HuggingFace `worstchan/EAT-*`), AST/SSAST (`timm`), SSAMBA (`mamba-ssm`) |
| Code | `src/birdclef/` |

### New Encoder Details

| Encoder | Source | Params | Pretraining | Architecture |
|---------|--------|--------|-------------|--------------|
| AST | `MIT/ast-finetuned-audioset-10-10-0.4593` | 87M | ImageNet → AudioSet (supervised) | ViT |
| SSAST | `yuangongfdu/ssast-base-patch-400` | 87M | AudioSet + LibriSpeech (SSL) | ViT |
| SSAMBA-tiny | [Google Drive](https://drive.google.com/drive/folders/1E1gf5SxdSByDJ16_WQvzTKn8lIoYtZiX) | 7M | AudioSet + LibriSpeech (SSL) | Mamba |
| SSAMBA-small | [Google Drive](https://drive.google.com/drive/folders/1E1gf5SxdSByDJ16_WQvzTKn8lIoYtZiX) | 26M | AudioSet + LibriSpeech (SSL) | Mamba |
| SSAMBA-base | [Google Drive](https://drive.google.com/drive/folders/1E1gf5SxdSByDJ16_WQvzTKn8lIoYtZiX) | 99M | AudioSet + LibriSpeech (SSL) | Mamba |

### Procedure

**Part A: ESC-50 (sanity check)**
1. Download ESC-50 from HuggingFace (`ashraq/esc50`)
2. Load pretrained checkpoints
3. Extract features, train linear probe
4. Evaluate with 5-fold CV (match published protocol)
5. Compare to published: BEATs 98.1%, EAT 95.9%

**Part B: AudioSet Bird Classes** (deferred)
1. Download AudioSet eval split (bird class IDs: `/m/015p6`, `/m/020bb7`, etc.)
2. Load same pretrained checkpoints
3. Extract features, train linear probe
4. Evaluate mAP on bird classes

**Part C: AST/SSAST/SSAMBA Benchmark (New)**
1. Download pretrained checkpoints (HuggingFace + Google Drive)
2. Implement encoder wrappers following existing pattern
3. Run linear probe on ESC-50 with 5-fold CV
4. Benchmark inference time (GPU and CPU)
5. Compare accuracy vs speed tradeoffs

### Variables

- **Independent**: Pretrained encoder (wav2vec2, BEATs, EAT, AST, SSAST, SSAMBA-tiny/small/base)
- **Dependent**: ESC-50 accuracy, inference time (ms/sample)
- **Controlled**: Classifier (linear), preprocessing (16kHz), evaluation protocol (5-fold CV)

## Evaluation

### Metrics

| Metric | Dataset | Description |
|--------|---------|-------------|
| Accuracy | ESC-50 | 5-fold CV accuracy (published standard) |
| mAP | AudioSet | Mean average precision |
| Inference time | Both | Forward pass time per clip (CPU) |

### Baseline

**ESC-50 (published results)**
| Model | Accuracy | Notes |
|-------|----------|-------|
| BEATs | 98.1% | Finetuned |
| EAT | 95.9% | Finetuned |
| AST | 95.6% | Finetuned |
| SSAST | ~88% | Linear probe |
| SSAMBA-base | 89.3% | Linear probe |
| SSAMBA-small | 85.5% | Linear probe |
| SSAMBA-tiny | 80.1% | Linear probe |
| Human | 81.3% | - |

**Inference Speed (published)**
| Model | vs SSAST |
|-------|----------|
| SSAMBA-tiny | 92.7% faster |
| SSAMBA-small | ~90% faster |
| SSAMBA-base | ~85% faster |

### Success Criteria

**Part A/B (Complete)**:
- ✅ ESC-50 results within 2% of published (validates setup)
- ✅ BEATs/EAT achieve >20% higher accuracy than wav2vec2

**Part C (New)**:
- **Confirm if**:
  1. SSAMBA-small achieves >80% ESC-50 accuracy (matches published 85.5%)
  2. SSAMBA inference is >5x faster than BEATs on CPU
  3. SSAMBA accuracy gap to BEATs is <10% (acceptable for speed tradeoff)
- **Reject if**: SSAMBA accuracy <75% or speed advantage <3x

## Limitations

- AudioSet has coarse bird labels (12 classes) - doesn't test fine-grained discrimination
- Linear probe may underestimate model capacity
- Some AudioSet clips may be unavailable (YouTube takedowns)

## AudioSet Bird Classes

```
/m/015p6    Bird (parent)
/m/020bb7   Bird vocalization, bird call, bird song
/m/07pggtn  Chirp, tweet
/m/07sx8x_  Squawk
/m/0h0rv    Pigeon, dove
/m/07r_25d  Coo
/m/04s8yn   Crow
/m/07r5c2p  Caw
/m/09d5_    Owl
/m/07r_80w  Hoot
/m/01dwxx   Gull, seagull
/m/05_wcq   Bird flight, flapping wings
```
