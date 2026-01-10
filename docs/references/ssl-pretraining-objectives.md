# Reference: Audio SSL Pretraining Objectives and Linear Probe Performance

**Source**: BEATs (arXiv:2212.09058), EAT (arXiv:2401.03497), Experiment 001
**Date Captured**: 2026-01-10
**Type**: deep-research

## Summary

Different self-supervised learning (SSL) pretraining objectives produce features with varying degrees of linear separability. Prediction-based methods (like BEATs) produce discriminative features that work well with linear probes, while reconstruction-based methods (like EAT) require finetuning to achieve optimal performance.

## Key Points

- BEATs uses discrete label prediction, making features inherently classification-ready
- EAT uses bootstrap reconstruction (data2vec 2.0 style), requiring finetuning for best results
- Linear probe performance gap: BEATs 2.3% vs EAT 18.0% below finetuned results
- For frozen feature extraction, prefer prediction-based SSL models

## Pretraining Objectives

### BEATs: Discrete Label Prediction

From the paper: BEATs uses "mask and label prediction manner" where the model learns to predict discrete semantic audio tokens.

Key design insight: "abstract the high-level audio semantics and discard the redundant details as in human perception"

The iterative process:
1. Random projection as initial acoustic tokenizer
2. Train audio SSL model to predict masked tokens
3. Distill new tokenizer from trained model
4. Repeat, refining both tokenizer and model

This creates a virtuous cycle where the model learns increasingly semantic representations.

### EAT: Bootstrap Self-Supervised Learning

From the paper: EAT is "inspired by data2vec 2.0 in image modality and Audio-MAE in audio modality"

Key components:
- **Utterance-Frame Objective (UFO)**: Enhances modeling of acoustic events
- **Large inverse block masks**: Critical for learning superior representations
- **Bootstrap paradigm**: Teacher-student representation matching

Optimizes for representation consistency rather than explicit classification.

## Linear Probe Performance Analysis

### Experimental Results (ESC-50, 5-fold CV)

| Encoder | Linear Probe | Finetuned (published) | Gap |
|---------|-------------|----------------------|-----|
| BEATs | 95.8% | 98.1% | 2.3% |
| EAT | 77.9% | 95.9% | 18.0% |
| wav2vec2 | 48.2% | N/A (speech model) | N/A |

### Why the Gap Differs

**BEATs (small gap)**:
- Pretraining objective is classification-like (predict discrete labels)
- Features are already linearly separable
- Linear probe is essentially the task it was trained for

**EAT (large gap)**:
- Pretraining objective is reconstruction/consistency
- Features are rich but entangled
- Requires nonlinear adaptation (finetuning) to separate classes

## Implications for BirdCLEF

### Frozen Feature Extraction
- **Use BEATs** when you need frozen features for downstream models
- Good for: prototypical networks, kNN classifiers, simple linear heads
- Advantage: No GPU needed for downstream training

### Finetuning Approach
- **EAT may catch up** with full finetuning
- Consider if you have labeled data and compute budget
- EAT claims 15x training speedup over other audio SSL methods

### Recommendation

For BirdCLEF 2026 with limited labeled data:
1. Start with BEATs frozen features + simple classifier
2. If more compute available, try finetuning EAT
3. Compare with bird-specific models (BirdNET, Perch)

## Connections

- **Related concepts**: [Audio encoders for bird classification]
- **Informs experiments**: `user/anthony/experiments/001-encoder-benchmark/`

## Citations

```bibtex
@inproceedings{chen2023beats,
  title={BEATs: Audio Pre-Training with Acoustic Tokenizers},
  author={Chen, Sanyuan and others},
  booktitle={ICML},
  year={2023}
}

@article{chen2024eat,
  title={EAT: Self-Supervised Pre-Training with Efficient Audio Transformer},
  author={Chen, Wenxi and others},
  journal={arXiv preprint arXiv:2401.03497},
  year={2024}
}
```

## Raw Notes

From BEATs paper abstract:
> "We propose BEATs, an iterative audio pre-training framework to learn Bidirectional Encoder representation from Audio Transformers, where an acoustic tokenizer and an audio SSL model are optimized by iterations."

From EAT paper:
> "We introduce a novel Utterance-Frame Objective (UFO) to enhance the modeling capability of acoustic events."

The key difference: BEATs optimizes for **semantic abstraction** while EAT optimizes for **representation efficiency**. Both achieve SOTA with finetuning, but BEATs wins for frozen features.
