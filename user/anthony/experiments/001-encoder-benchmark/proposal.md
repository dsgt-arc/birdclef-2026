# Proposal: Foundation Model Encoder Benchmark

**Status**: in-progress
**Created**: 2025-01-09
**Author**: anthony

## Related Documentation

- **Concepts**:
  - [LoRA Finetuning for Audio Transformers](../../../docs/concepts/lora-finetuning-audio.md) - Parameter-efficient finetuning approach
- **References**:
  - [SSL Pretraining Objectives](../../../docs/references/ssl-pretraining-objectives.md) - Why prediction-based SSL (BEATs) outperforms reconstruction-based (EAT) for linear probes
  - [BEATs Architecture](../../../docs/references/deep-research/beats-architecture.md) - Deep dive into BEATs model
  - [Wav2Vec2 Landscape](../../../docs/references/deep-research/wav2vec2-landscape.md) - Speech foundation model context

## Hypothesis

BEATs and EAT will significantly outperform wav2vec2 on audio classification because they were pretrained on AudioSet, while wav2vec2 was pretrained on speech (LibriSpeech). We expect to match published results on ESC-50 (sanity check) and see similar patterns on AudioSet bird classes.

## Background

Before exploring distillation strategies, we need baseline performance numbers for candidate encoders. Using AudioSet's bird classes eliminates domain shift as a confounding variable - we're measuring pure encoder quality on the exact distribution they were trained on.

This informs our choice of teacher model for distillation.

## Method

### Approach

Evaluate frozen feature extraction from three pretrained encoders on two benchmarks:
1. **ESC-50** - Standard benchmark, sanity check against published results
2. **AudioSet bird classes** - Domain-relevant, zero pretraining shift

Use linear probe to isolate encoder quality.

### Setup

| Component | Details |
|-----------|---------|
| Data | ESC-50 (2k clips, 50 classes), AudioSet bird subset (~27k clips, 12 classes) |
| Compute | Single GPU, ~1-2 hours per model per dataset |
| Dependencies | `transformers`, BEATs (`vendor/unilm/beats`), EAT (`vendor/EAT`) |
| Code | `src/birdclef/` |

### Procedure

**Part A: ESC-50 (sanity check)**
1. Download ESC-50 from HuggingFace (`ashraq/esc50`)
2. Load pretrained checkpoints
3. Extract features, train linear probe
4. Evaluate with 5-fold CV (match published protocol)
5. Compare to published: BEATs 98.1%, EAT 95.9%

**Part B: AudioSet Bird Classes**
1. Download AudioSet eval split (bird class IDs: `/m/015p6`, `/m/020bb7`, etc.)
2. Load same pretrained checkpoints
3. Extract features, train linear probe
4. Evaluate mAP on bird classes

### Variables

- **Independent**: Pretrained encoder (wav2vec2, BEATs, EAT)
- **Dependent**: mAP on AudioSet bird classes
- **Controlled**: Classifier (linear), preprocessing (16kHz, 10s clips)

## Evaluation

### Metrics

| Metric | Dataset | Description |
|--------|---------|-------------|
| Accuracy | ESC-50 | 5-fold CV accuracy (published standard) |
| mAP | AudioSet | Mean average precision |
| Inference time | Both | Forward pass time per clip (CPU) |

### Baseline

**ESC-50 (published results)**
| Model | Accuracy |
|-------|----------|
| BEATs | 98.1% |
| EAT | 95.9% |
| Human | 81.3% |

**AudioSet**
- Random: ~0.08 mAP
- CNN14 (PANNs): ~0.35 mAP

### Success Criteria

- **Confirm if**:
  1. ESC-50 results within 2% of published (validates setup)
  2. BEATs/EAT achieve >20% higher mAP than wav2vec2 on bird classes
- **Reject if**: wav2vec2 matches AudioSet-pretrained models, or ESC-50 results far below published

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
