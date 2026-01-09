# Handling Soundscape Domain Shift

The core challenge in BirdCLEF: models train on clean focal recordings but must classify noisy soundscape recordings at test time. This document surveys approaches and practical recommendations.

## The Domain Gap Problem

**Focal recordings** (training):
- Single target species, close microphone
- Clean audio, high SNR
- Controlled recording conditions

**Soundscape recordings** (test):
- Multiple overlapping species
- Environmental noise (wind, rain, insects, traffic)
- Distant microphones, variable SNR
- Real-world deployment conditions

This gap causes significant performance degradation. Models overfit to "clean" acoustic patterns that don't exist in the wild.

## Approaches

### 1. Source Separation / Denoising

**MixIT (Mixture Invariant Training)**
- Google's unsupervised source separation for bird audio
- Trained specifically on bird recordings: 10.5 dB SNR improvement
- Generic AudioSet model: only 4.4 dB improvement
- Available in bioacoustics-model-zoo as `SeparationModel`

```python
import bioacoustics_model_zoo as bmz
model = bmz.SeparationModel(
    checkpoint='path/to/bird_mixit_model_checkpoints/output_sources4/model.ckpt'
)
separated = model.separate_audio(audio)
```

**Limitations**:
- Adds inference latency (~100-200ms per clip)
- Hard to qualitatively evaluate denoising quality
- May introduce artifacts or remove legitimate bird sounds
- Requires downloading large checkpoint

**Verdict**: Potentially useful for training data augmentation, risky for test-time preprocessing without careful validation.

Sources:
- [Google blog: Separating Birdsong in the Wild](https://research.google/blog/separating-birdsong-in-the-wild-for-classification/)
- [Paper: Improving Bird Classification with Unsupervised Sound Separation](https://arxiv.org/abs/2110.03209)

### 2. Domain-Invariant Training

**ProtoCLR Approach**
- Prototypical Contrastive Learning for domain invariance
- Key insight: learn representations invariant to recording conditions
- Trains on focal (Xeno-Canto), evaluates on soundscape
- 19M parameter CvT-13 matches larger models on few-shot soundscape tasks

**Why it works**:
- Contrastive learning pulls same-species embeddings together regardless of background
- Prototype-based classification is robust to distribution shift
- Focus on discriminative features, not recording artifacts

**Verdict**: Most practical approach. Already implemented in ProtoCLR, can adopt training methodology.

Source: [ProtoCLR paper](https://arxiv.org/abs/2409.08589)

### 3. Data Augmentation

**BirdSet's Approach**
- Mixes focal recordings with background noise from DCASE18
- Downloads background noise via `resources/utils/download_background_noise.py`
- Applied during training with configurable intensity

**Common augmentations**:
- Background noise mixing (SNR 5-20 dB)
- SpecAugment (time/frequency masking)
- Time stretching, pitch shifting
- Mixup across recordings (BirdSet's GeneralizedMixup)

**Verdict**: Essential baseline. Easy to implement, well-supported in BirdSet.

Source: [BirdSet repository](https://github.com/DBD-research-group/BirdSet)

### 4. Test-Time Adaptation

**Perch's SFDA (Source-Free Domain Adaptation)**
- Adapts model to target domain without source data access
- NOTELA method from ICML 2023
- Processes multiple test samples to estimate domain shift

**Kaggle constraint**: Files may need to be processed independently, making transductive methods difficult. DCASE 2024 explicitly required this:
> "strategy sidesteps the use of transductive learning to ensure compliance"

**Verdict**: Likely not Kaggle-compatible unless processing is purely per-file.

Source: [Google blog: Source-Free Domain Adaptation](https://research.google/blog/in-search-of-a-generalizable-method-for-source-free-domain-adaptation/)

### 5. Negative Hard Sampling

**DCASE 2024 Approach**
- Mine hard negatives: sounds similar to target but not the species
- Helps distinguish bird calls from confusable environmental sounds
- Achieved 52.14% on DCASE 2024 Task 5 (bioacoustic few-shot detection)

**Implementation**:
- During training, find samples with high model confidence but wrong label
- Add these as explicit negatives
- Forces model to learn finer-grained distinctions

**Verdict**: Worth exploring if baseline performance plateaus.

Source: [Mind the Domain Gap (DCASE 2024)](https://arxiv.org/abs/2403.18638)

## Practical Recommendations

### For BirdCLEF Competition

| Priority | Approach | Effort | Expected Impact |
|----------|----------|--------|-----------------|
| 1 | BirdSet background noise augmentation | Low | High |
| 2 | ProtoCLR-style contrastive training | Medium | High |
| 3 | Ensemble clean + augmented models | Low | Medium |
| 4 | MixIT on training data (careful) | Medium | Unknown |
| 5 | Negative hard sampling | Medium | Medium |
| 6 | Test-time adaptation | High | Maybe not allowed |

### Recommended Pipeline

**Training**:
```
1. Start with BirdSet data loading + background noise augmentation
2. Use ProtoCLR training objectives (or similar contrastive loss)
3. Validate on soundscape eval sets (HSN, SSW, PER, etc.)
4. Optionally: MixIT-augment a subset of training data, compare
```

**Inference**:
```
1. Standard preprocessing (mel spectrogram)
2. Model prediction
3. Optional: ensemble with model trained under different augmentation
```

### What NOT to Do

- Don't rely solely on clean focal training data
- Don't apply MixIT at test time without validation
- Don't assume transductive methods will be allowed
- Don't ignore the domain gap and hope the model generalizes

## Open Questions

1. **MixIT quality**: How to evaluate whether separation helps or hurts for specific recordings?
2. **Augmentation intensity**: What SNR range for background mixing is optimal?
3. **ProtoCLR vs supervised**: Does contrastive pre-training beat supervised on soundscape transfer?
4. **Ensemble weighting**: How to combine focal-trained and soundscape-augmented models?

## References

- [Mind the Domain Gap: Systematic Analysis on Bioacoustic Sound Event Detection](https://arxiv.org/abs/2403.18638) (DCASE 2024)
- [Improving Bird Classification with Unsupervised Sound Separation](https://arxiv.org/abs/2110.03209) (Google, 2021)
- [ProtoCLR: Domain-Invariant Representation Learning of Bird Sounds](https://arxiv.org/abs/2409.08589) (2024)
- [Perch SFDA: Source-Free Domain Adaptation](https://research.google/blog/in-search-of-a-generalizable-method-for-source-free-domain-adaptation/) (ICML 2023)
- [BirdSet: A Multi-Task Benchmark for Classification in Avian Bioacoustics](https://arxiv.org/abs/2403.10380) (NeurIPS 2024)
