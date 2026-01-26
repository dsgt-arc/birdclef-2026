# Reference: BirdSet - A Large-Scale Dataset for Audio Classification in Avian Bioacoustics

**Source**: arXiv:2403.10380
**Date Captured**: 2026-01-24
**Relevance**: Establishes baseline benchmark for experiment 001 (BirdSet Bioacoustics Model Benchmark)

---

## Summary

BirdSet introduces a comprehensive benchmark designed to standardize and advance machine learning research in bird sound classification. It addresses the lack of a unified, large-scale benchmark (like ImageNet or AudioSet) for avian bioacoustics.

## Dataset Composition

### Training Data
- **Size**: Over 6,800 hours of audio
- **Species**: Nearly 10,000 bird species
- **Source**: Primarily Xeno-Canto focal recordings
- **Format**: High-quality recordings where the bird is the main subject

### Evaluation Data
- **Size**: Over 400 hours of strongly labeled soundscape recordings
- **Labels**: Start and end times of calls are annotated
- **Source**: 8 distinct monitoring projects
- **Purpose**: Test generalization to real-world environmental recordings

## The 8 Evaluation Datasets

| Code | Name | Region | Notes |
|------|------|--------|-------|
| **HSN** | High Sierra Nevada | California, USA | Mountain ecosystem |
| **PER** | Amazon Basin | Peru | Tropical rainforest |
| **NES** | Colombia & Costa Rica | Central/South America | Neotropical |
| **UHH** | Hawaiian Islands | Hawaii, USA | Island ecosystem, endemic species |
| **POW** | Powdermill Nature Reserve | Pennsylvania, USA | Temperate forest |
| **SSW** | Sapsucker Woods | New York, USA | Cornell Lab site |
| **SNE** | Sierra Nevada | Spain | European mountain ecosystem |
| **NBP** | NIPS4BPlus | Various | Standard bioacoustics benchmark |

## Training Scenarios

| Scenario | Description | Use Case |
|----------|-------------|----------|
| **XCL** | Xeno-Canto Large | Full large-scale training dataset |
| **XCM** | Xeno-Canto Medium | Resource-constrained scenarios |
| **DT** | Dedicated Training | Fine-tuning for specific target test set species |

## Benchmark Challenge: Covariate Shift

The core challenge is **domain shift**:
- **Training**: High-quality focal recordings (bird is main subject)
- **Testing**: Noisy soundscapes (real-world passive monitoring, distant birds, environmental noise)

This tests whether models can generalize from clean training data to realistic deployment conditions.

## Benchmarked Models in Paper

The paper establishes baselines with six architectures:

| Model | Type | Notes |
|-------|------|-------|
| EfficientNet (B1, B3) | CNN | Standard vision backbone adapted for spectrograms |
| ConvNeXt | CNN | Modern CNN architecture |
| Wav2Vec2 | Transformer | Audio-specific, self-supervised pretraining |
| AST | Transformer | Audio Spectrogram Transformer |
| EAT | Transformer | Event Audio Transformer |
| Bird-MAE | Autoencoder | Masked Autoencoder for self-supervised learning |

## Key Findings

1. **Generalization is Hard**: Models trained on clean focal recordings struggle with domain shift to noisy soundscapes
2. **Pre-training Value**: BirdSet pre-training + fine-tuning outperforms training from scratch or generic audio pre-training
3. **Class Imbalance**: Some species have very few training examples, affecting per-class performance

## Evaluation Metrics

- **cmAP (Class-mean Average Precision)**: Primary metric, macro-averaged across classes
- **Per-class AP**: Identifies which species benefit from transfer learning
- **Handles class imbalance**: Treats all classes equally regardless of sample count

## Resources

- **HuggingFace Dataset**: `DBD-research-group/BirdSet`
- **Code**: Available on GitHub
- **Pre-trained Weights**: Available on HuggingFace

## Implications for Experiment 001

### Current Scope (HSN only)
The current proposal uses HSN as a pilot dataset. This is reasonable for initial validation but doesn't capture the full benchmark.

### Recommended Expansion
To properly reproduce BirdSet results, we should evaluate on **all 8 test datasets**:
1. Start with HSN (pilot) - currently planned
2. Expand to remaining 7 datasets after validating pipeline
3. Report results comparable to paper baselines

### Model Comparison
The paper benchmarks CNN and Transformer models. Our experiment uses:
- **BirdNET**: Not in original paper (external pre-trained model)
- **Perch**: Not in original paper (external pre-trained model)

This provides complementary results showing how domain-specific pre-trained models compare to architectures trained directly on BirdSet.

---

## Related Documents

- [Experiment 001 Proposal](../../user/murilo/experiments/001-birdset-bioacoustics-benchmark/proposal.md)
- [Experiment 001 Design](../../user/murilo/experiments/001-birdset-bioacoustics-benchmark/design.md)
- [BirdSet Vendor Doc](../vendor/birdset.md)
