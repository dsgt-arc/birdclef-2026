# Concept: Source Separation for Smart Augmentation

**Status**: exploratory/future-work seed
**Created**: 2026-03-14
**Updated**: 2026-03-14

## Idea

Use audio source separation (Bird MixIT, biodenoising) to decompose recordings into foreground (bird vocalizations) and background (environmental noise), then recombine them in controlled ways to create realistic training augmentations that bridge the focal-to-soundscape domain gap. This goes beyond simple noise mixing by leveraging species co-occurrence frequencies and distance simulation to create ecologically plausible synthetic soundscapes.

The core pipeline: **separate → manipulate → remix**. Biodenoising provides a foreground/background split (1 clean output + residual noise). MixIT provides multi-source decomposition (4-8 channels of individual bird voices + noise). Together, they enable fine-grained control over the acoustic scene composition.

## Motivation

BirdCLEF 2026 introduces soundscape recordings alongside focal recordings. The fundamental domain gap remains: models train on clean focal recordings (single target species, close mic, high SNR) but must classify noisy soundscapes (multiple overlapping species, distant mics, environmental noise). Previous experiments (exp 014, 015) showed that soft labels and SED heads didn't help — the bottleneck is representation quality under domain shift, not label or architecture issues.

Source separation offers a principled way to decompose and recompose audio scenes, enabling:
1. **Controlled SNR augmentation**: Rather than adding random noise, use actual separated backgrounds from soundscapes
2. **Co-occurrence-aware mixing**: Create multi-species mixtures that reflect real ecological communities
3. **Distance simulation**: Adjust loudness of individual separated sources + apply low-pass filtering to simulate varying recording distances
4. **Domain adaptation without domain labels**: No need for paired focal/soundscape examples

## Background

### Source Separation Tools

**Bird MixIT** (Google, 2021):
- TDCN++ architecture, trained on 704 Xeno-Canto species
- 4 or 8 output sources per clip, 22.05 kHz, TensorFlow
- 10.5 dB SI-SNR improvement (bird-specific) vs 4.4 dB (generic AudioSet). Note: SI-SNR and SI-SDR are mathematically equivalent.
- Max-channel strategy at inference: classify all channels + original, take max probability per species
- Downstream improvement (cmAP): +6.3% on High Sierras (0.527→0.560), +2.1% on Caples (0.334→0.341), +0.7% on Sapsucker Woods (0.304→0.306)
- See: [docs/vendor/bird-mixit.md](../vendor/bird-mixit.md)

**Biodenoising** (Earth Species Project, 2024):
- Demucs dns48 architecture, distillation-based training with pseudo-clean targets from speech models
- Single denoised output (foreground), background = original - denoised
- 12.85 dB SI-SDR (best variant, Demucs + noisereduce pseudo-clean targets), 16 kHz, PyTorch
- Fast domain adaptation: 5-10 min of target audio, few minutes training
- Multi-taxa: birds, amphibians, insects, mammals (all BirdCLEF+ taxa)
- See: [docs/vendor/biodenoising.md](../vendor/biodenoising.md)

**RemixIT** (Tzinis et al., 2022):
- Self-supervised domain adaptation for separation: teacher model on out-of-domain data generates pseudo-targets, permute/remix estimated clean+noise to create bootstrapped training data for student
- Directly applicable to focal-to-soundscape adaptation: train teacher on focal (cleaner), adapt to soundscape without labels
- Paper: [arXiv:2202.08862](https://arxiv.org/abs/2202.08862)

**Mix2 — Mixture of Mixups for Anuran Sounds** (2024):
- Combines Mixup, Manifold Mixup, and MultiMix for multi-label bioacoustics
- Future work explicitly proposes integrating eBird temporal/spatial co-occurrence constraints
- Shows effectiveness for rare classes and multi-label scenarios
- Paper: [arXiv:2403.09598](https://arxiv.org/abs/2403.09598)

### Prior Art in Competition Solutions

**No top BirdCLEF solution (2021-2025) has used source separation.** The dominant technique is iterative pseudo-labeling (3-4 rounds). Standard augmentations: SpecAugment, mixup, background noise from DCASE18. The Bird MixIT paper itself noted "applying separation to classifier training data may improve classifier quality" as future work but didn't implement it. This means source separation for augmentation is an **unexplored competitive advantage** — or it may be too expensive for marginal gain.

Google's Bird MixIT paper found that **random low-pass filtering** during training (simulating distance) helped for identifying species from the High Sierras. This is a simpler version of the distance simulation idea.

### Connection to Sequence Modeling (SSAMBA / Mamba)

Source separation creates multiple parallel streams per recording. For a Mamba/SSM architecture processing long sequences:
- **Co-occurrence attention**: Species that frequently co-occur in soundscapes create natural attention patterns. A model that has seen realistic multi-species mixtures during training can learn to attend to temporal co-occurrence patterns in the sequence.
- **Foreground/background disentanglement**: Mamba's selective state space mechanism could learn to route foreground and background through different state channels, analogous to what source separation does explicitly.
- **CPU viability**: SSAMBA's linear-time complexity makes it viable for CPU inference. Source separation is training-only (expensive), but the representations learned from augmented data carry over to inference.
- **Mamba for separation itself**: MSNet (2025) and SPMamba use dual-path Mamba for music/speech separation with O(L) complexity. U-Mamba-Net achieves 8.50 dB SI-SNR on Libri2Mix with only 4.4M params. A Mamba-based separator could be more efficient than TDCN++.
- **BioMamba** ([arXiv:2512.03563](https://arxiv.org/abs/2512.03563)): Mamba-based bioacoustics classifier that matches transformer AVES with less VRAM and better data efficiency. Validates Mamba is viable for bioacoustics at scale.
- **Cleaner input sequences**: Separated sources let Mamba's selective SSM focus on one species' temporal pattern at a time, rather than disentangling a mixture internally.

## Proposed Augmentation Strategies

### Strategy A: Foreground/Background Remix (Biodenoising)

```
focal_recording → biodenoising → foreground (vocalization) + background (noise)
soundscape_recording → biodenoising → foreground (vocalizations) + background (noise)

augmented = focal_foreground + alpha * soundscape_background
# alpha controls SNR, sampled from realistic soundscape SNR distribution
```

- Simple, fast, uses only biodenoising (PyTorch)
- Creates focal birds in soundscape noise environments
- Domain adaptation: fine-tune biodenoising on soundscape data first

### Strategy B: Multi-Source Co-occurrence Mixing (MixIT)

```
focal_recording → MixIT → [source0, source1, source2, source3]
# Identify which source(s) contain the target species (by energy or classifier)

co_occurring_species = lookup_cooccurrence(target_species, metadata)
co_focal_recording → MixIT → [sources...]
# Select source containing co-occurring species

augmented = target_source + co_occurring_source * beta + noise_sources * gamma
# beta, gamma control relative loudness (distance simulation)
```

- Ecologically plausible: species that co-occur in nature appear together in training
- Requires species co-occurrence data (from eBird, BirdCLEF metadata, or recording-level labels)
- More compute: MixIT on all focal recordings + co-occurrence lookup

### Strategy C: Distance-Aware Loudness Manipulation

For each separated source:
1. Adjust gain (loudness → distance proxy): -6 dB per doubling of distance
2. Apply random low-pass filter (high frequencies attenuate faster with distance)
3. Add slight reverb (distant sources have more environmental reflection)

```python
def simulate_distance(source, distance_factor):
    """Simulate recording a source at distance_factor * original distance.

    NOTE: The gain formula follows ideal free-field inverse square law.
    Real forest/environment attenuation is typically greater due to
    scattering, absorption, and ground effects. The cutoff_hz formula
    is a rough heuristic (not physics-based); real atmospheric absorption
    follows ISO 9613-1 which is frequency-dependent and nonlinear.
    """
    gain_db = -6 * np.log2(distance_factor)  # inverse square law (ideal)
    cutoff_hz = 8000 / distance_factor       # rough heuristic for high-freq attenuation
    source = apply_gain(source, gain_db)
    source = lowpass_filter(source, cutoff_hz)
    return source
```

### Strategy D: Soundscape-to-Focal Adaptation

Reverse direction: make soundscapes more like focal recordings for test-time benefit.

```
soundscape → MixIT → [sources]
# For each source, classify species
# Select the target species source → now have a "pseudo-focal" recording
# Use as additional training data with high-confidence pseudo-labels
```

This connects to pseudo-labeling (the dominant 2025 technique) but with cleaner per-species separation.

## Computational Considerations

### Training Pipeline Impact

MixIT produces 4-8 sources per clip, effectively multiplying data streams:
- The "939K clips" figure comes from BirdNET's 3s/1s windowing (exp 010), not raw recordings. MixIT uses 6s windows natively, so clip segmentation may differ.
- 939K clips × 4 sources = **3.76M** audio files to store/process
- Storage depends on sample rate: at MixIT's native 22.05 kHz (3s clips): ~496 GB; if upsampled back to 32 kHz: ~720 GB
- Separation is a one-time offline cost, but significant. Rough GPU estimate: ~13-50 hours on RTX 6000 (feasible as 1-2 sbatch jobs).

**Recommendation**: Pre-compute separations offline (sbatch job), store separated sources on scratch storage, load during training.

### Biodenoising vs MixIT Cost

| Operation | Biodenoising | Bird MixIT |
|-----------|-------------|-----------|
| Framework | PyTorch (easy integration) | TensorFlow (separate env) |
| Outputs per clip | 2 (foreground + background) | 4-8 sources |
| Storage multiplier | 2x | 4-8x |
| Domain adaptation | Yes (fast, 5-10 min) | No |
| Multi-taxa support | Yes (birds, amphibians, insects, mammals) | Birds only |

**For BirdCLEF+ 2026** (206 species including non-birds): biodenoising may be more appropriate because it handles all taxa. MixIT is bird-specific.

### On-the-Fly vs Pre-computed

Pre-computed (recommended):
- Run separation as batch job on scratch storage
- Load separated sources during training
- No GPU overhead during training

On-the-fly (not recommended):
- Adds 100-200ms latency per clip (GPU) during training
- MixIT requires TensorFlow alongside PyTorch
- 4-8x more data to process through the model

## Co-occurrence Data Sources

For strategies B and C, we need species co-occurrence data:

1. **BirdCLEF metadata**: Training CSV has `latitude`, `longitude`, `date` per recording. Group by location/date to find co-occurring species.
2. **eBird Basic Dataset (EBD)**: Tab-separated file with all validated sightings. Group by `SAMPLING EVENT IDENTIFIER` to build co-occurrence matrix. Request at [ebird.org/science](https://ebird.org/science/use-ebird-data/download-ebird-data-products). Note: access approval can take weeks.
3. **Kaggle notebook**: [Species Co-Occurrence from eBird Checklists](https://www.kaggle.com/code/lobrien/species-co-occurrence-from-ebird-checklists) — directly builds NxN co-occurrence matrices.
4. **Exp 010 teacher labels**: BirdNET predictions on all recordings contain implicit co-occurrence (which species BirdNET detects in the same recording).
5. **eBird Status Data Products**: Species distributions on 3km grid, 52 weeks/year, 2980 species. Can derive temporal co-occurrence from overlapping ranges.

## Open Questions

- **Separation artifacts as training confound**: Imperfect separation introduces artifacts (spectral holes, musical noise, over-separation fragments). If training data includes these artifacts but test data does not go through separation, the model may learn artifact-specific features that hurt generalization. This is the single biggest risk of the approach.
- **Does foreground/background remixing actually improve cmAP on soundscapes?** This is the core hypothesis — must test empirically.
- **What SNR distribution matches real soundscapes?** Need EDA on BirdCLEF 2026 soundscape recordings to characterize typical SNR ranges.
- **How to identify which MixIT channel contains the target species?** Simple energy heuristic, or need a classifier in the loop?
- **Does co-occurrence mixing help or hurt?** Adding co-occurring species may confuse the model if labels don't reflect the added species.
- **Biodenoising CC-BY-NC license**: Is Kaggle competition use considered "non-commercial"? The code is MIT, so we could retrain on our own data to avoid the weight license issue.
- **MixIT TensorFlow dependency**: Can we convert the TF checkpoint to ONNX or TorchScript for easier pipeline integration?
- **How much separation quality is needed for augmentation to help?** Even imperfect separation might provide useful diversity.

## Testable Hypotheses

1. **H1 (Foreground remix)**: Training with biodenoising-separated focal birds remixed into soundscape backgrounds improves cmAP on soundscape evaluation by >5% over noise-only augmentation (BirdSet DCASE18 backgrounds).

2. **H2 (Distance simulation)**: Adding low-pass filtered + gain-reduced copies of focal recordings (simulating distance) improves cmAP on distant/quiet birds by >10%.

3. **H3 (Co-occurrence mixing)**: Multi-species mixtures that respect ecological co-occurrence produce higher cmAP than random species mixing (standard mixup).

4. **H4 (MixIT max-channel)**: Applying MixIT at inference time with max-channel aggregation improves cmAP on soundscape data. The paper found +0.7% to +6.3% relative cmAP depending on the dataset (SSW: +0.7%, CAP: +2.1%, HSN: +6.3%). This is a replication study — expect dataset-dependent gains.

5. **H5 (SSAMBA benefit)**: Sequence models (SSAMBA) benefit more from co-occurrence augmentation than CNNs (EfficientNet), because they can model temporal co-occurrence patterns. (Speculative — no prior evidence for this specific claim; grounded in the architectural difference between sequential and convolutional processing.)

## Connections

- **Based on**: [Bird MixIT](../vendor/bird-mixit.md), [Biodenoising](../vendor/biodenoising.md), [Soundscape Domain Shift](soundscape-domain-shift.md)
- **Related to**: [Weak Supervision Label Fusion](weak-supervision-label-fusion.md) (co-occurrence data from teacher labels), [Data and Compute Scale](data-and-compute-scale.md) (storage/compute requirements)
- **Informed by**: BirdCLEF 2025 Kaggle solutions (pseudo-labeling, SED, temporal smoothing)
- **Could lead to**: Experiment proposal for source-separation augmentation pipeline, integrated with exp 009 (SSAMBA pretrain) or exp 012 (EfficientNet) for evaluation
- **SSAMBA connection**: Sequence modeling benefits from learning co-occurrence patterns that mimic real soundscape temporal structure

## Notes

- The Bird MixIT paper notes that **separation-only classification degrades performance**. Always include the original mixture as an additional channel. For augmentation, this means the original recording should remain in the training set alongside augmented versions.
- Biodenoising's pseudo-clean approach tried MixIT-style unsupervised separation internally and found it "did not generalize well" — the two tools solve different problems (denoising vs separation) and should be used for their respective strengths.
- BirdCLEF+ 2026 includes non-bird taxa (amphibians, insects, mammals). Biodenoising handles these; Bird MixIT does not. Consider biodenoising as the primary tool, with MixIT as an optional add-on for bird-specific multi-source decomposition.
- The 4-8x data expansion from MixIT is a training-only cost. At inference time, the classifier runs on the original audio (or optionally on separated channels with max-channel aggregation, but this multiplies inference cost too).
- Co-occurrence data sources: BirdCLEF metadata (recording-level labels), eBird checklist data, or compute from our own teacher labels (exp 010: BirdNET predictions on all recordings contain implicit co-occurrence information).
