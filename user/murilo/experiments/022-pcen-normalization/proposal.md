---
experiment: "022-pcen-normalization"
status: proposed
date: 2026-04-06
author: Murilo Gustineli
---

# Experiment 022: PCEN Normalization — Closing the Domain Gap

## Hypothesis

Replacing AmplitudeToDB normalization with PCEN (Per-Channel Energy
Normalization) will improve LB score by producing mel representations that
are invariant to recording conditions, directly addressing the domain shift
between clean focal recordings and noisy Pantanal soundscapes. PCEN's
adaptive gain control should produce more consistent features across
domains, unlike AmplitudeToDB which applies a fixed dB transform that
preserves absolute energy differences between recording setups.

## Background

### Pattern analysis motivates this experiment

Cross-experiment analysis of 40+ model variants (exps 006–021) revealed a
clear pattern: **the only changes that improved LB were input representation
changes and post-processing**. Every training recipe change (loss functions,
augmentation strategies, head architectures, regularization — 14 attempts)
regressed or was flat on LB. See
[`notes/2026-04-06-experiment-pattern-analysis.md`](../../notes/2026-04-06-experiment-pattern-analysis.md).

The domain shift between focal recordings (clean, single-species, variable
recording equipment/volume) and Pantanal soundscapes (noisy, multi-species,
consistent recorder) is our fundamental bottleneck. Models that improve on
focal-like validation data consistently hurt on soundscape-like test data.

### AmplitudeToDB was our second-largest gain

| Change | LB Delta | Experiment |
|--------|----------|------------|
| HGNetV2-B0 backbone | +0.037 | exp 019 v5g |
| **AmplitudeToDB norm (on EffNet)** | **+0.034** | **exp 017 v4b** |
| **AmplitudeToDB + NNAudio (on HGNet)** | **+0.014** | **exp 019 v5j** |
| SED architecture | +0.028 | exp 016 |

AmplitudeToDB was responsible for +0.014 to +0.034 LB — our second-largest
gain after the backbone swap. This confirms that **how we normalize the mel
spectrogram has an outsized impact on LB**. PCEN is a more principled version
of the same idea, specifically designed for the variable-SNR conditions that
cause our domain shift.

### What is PCEN?

PCEN (Wang et al. 2017, "Trainable Frontend for Robust and Far-Field Keyword
Spotting") applies per-frequency-band adaptive gain normalization:

```
PCEN(E) = (E / (ε + M)^α + δ)^r - δ^r
```

Where:
- `E` is the mel spectrogram energy
- `M` is a smoothed version of `E` (IIR filter with time constant `s`)
- `α` controls automatic gain control strength (AGC)
- `δ` is a bias term (root compression offset)
- `r` is the compression exponent (dynamic range compression)
- `ε` is a small constant for numerical stability

Key properties:
1. **Adaptive gain control**: dividing by smoothed energy `M^α` normalizes
   each frequency band relative to its recent energy, suppressing stationary
   background noise while preserving transient events (vocalizations)
2. **Learnable parameters**: `s`, `α`, `δ`, `r` can be trained end-to-end
   via backpropagation, adapting to the specific domain
3. **Recording-invariant**: a bird call at high gain in a focal recording
   and the same call at low SNR in a soundscape produce more similar features
   after PCEN than after AmplitudeToDB
4. **Bioacoustics standard**: used in Google's Perch (our best model at
   0.925 LB), BirdNET, and many bioacoustics research papers

### Why PCEN addresses our domain shift

AmplitudeToDB applies `10 * log10(mel + offset)` — a fixed nonlinear
transform that preserves absolute energy relationships. A loud focal
recording and a quiet soundscape recording of the same species produce
very different feature magnitudes.

PCEN normalizes relative to the local energy in each frequency band. The
automatic gain control (AGC) component `E / M^α` acts like a noise gate
that adapts to each recording's noise floor. This makes features from
different recording conditions (equipment, distance, SNR) more comparable.

The domain shift we observe (training on focal recordings, testing on
soundscapes) is largely a recording-condition mismatch. PCEN is
specifically designed to handle this.

## Method

### Approach

Replace the AmplitudeToDB normalization step in our mel pipeline with PCEN,
keeping everything else from the v5j recipe (HGNetV2-B0, LSEHead, BCE,
OneCycleLR, 20 epochs). Test both learnable and fixed PCEN parameters.

PyTorch implementation: https://github.com/daemon/pytorch-pcen

### Setup

| Component | Details |
|-----------|---------|
| Data | Same as v5j: 32K focal + 653 labeled soundscape segments, 234 classes |
| Compute | PACE GPU (RTX-6000 or V100), ~40 min per run |
| Dependencies | pytorch-pcen (or custom PCEN layer), NNAudio for base mel |
| Code | Branch from main, new config in `022-pcen-normalization/config/` |
| Baseline | v5j (0.906 LB) — HGNetV2-B0, AmplitudeToDB norm |

### Procedure

1. **Implement PCEN layer**: integrate pytorch-pcen as an `nn.Module` that
   replaces the AmplitudeToDB step in the GPU mel pipeline. The PCEN layer
   sits between `MelSpectrogram` output and the backbone input.

2. **v10a — Learnable PCEN**: PCEN with trainable parameters (s, α, δ, r)
   initialized to standard values (s=0.025, α=0.98, δ=2.0, r=0.5). Parameters
   learn end-to-end with the model. Everything else identical to v5j.

3. **v10b — Fixed PCEN**: PCEN with frozen parameters (same initial values,
   `requires_grad=False`). Controls for whether learnable PCEN overfits to
   focal data characteristics.

4. **v10c — PCEN + AmplitudeToDB**: stack both — AmplitudeToDB followed by
   PCEN. Tests whether PCEN adds value on top of our existing normalization
   rather than replacing it.

5. **ONNX export + LB submission**: export best variant via exp 010 pipeline.

### Variables

- **Independent**: mel normalization method (AmplitudeToDB vs learnable PCEN
  vs fixed PCEN vs PCEN+AmplitudeToDB)
- **Dependent**: LB score (row-wise AUROC), offline val/auroc, val/cmap
- **Controlled**: backbone (HGNetV2-B0), head (LSEHead), loss (BCE),
  scheduler (OneCycleLR), epochs (20), data split (90/10), augmentation
  (spectrogram MixUp), mel parameters (n_fft=2048, n_mels=256, hop=313)

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| LB (row-wise AUROC) | Primary — only reliable metric per pattern analysis |
| val/auroc | Secondary — offline, known to be anti-correlated with LB |
| val/cmap | Secondary — class-mean average precision |

### Baseline

| Model | val/auroc | LB | Normalization |
|-------|-----------|-----|---------------|
| v5j (baseline) | 0.874 | **0.906** | AmplitudeToDB |
| v5g (no AmpToDB) | 0.872 | 0.892 | ttahara min-max |

### Success Criteria

- **Confirm if**: any PCEN variant scores ≥0.910 LB (≥+0.004 over v5j),
  demonstrating that adaptive normalization improves soundscape generalization
- **Reject if**: all PCEN variants score ≤0.906 LB, suggesting AmplitudeToDB
  already captures the normalization benefit and PCEN adds no value for our
  backbone/recipe combination
- **Partial success**: if PCEN matches v5j on LB but improves cmap (per-class
  performance), it may still be valuable for rare species / soundscape-only
  species where domain shift is most acute

## Limitations

- PCEN's IIR smoothing filter has a time constant that assumes temporal
  continuity. Our 5s training windows may be too short for the AGC to adapt
  properly (especially at segment boundaries). Inference on 60s soundscapes
  is fine — the issue is during training.
- The pytorch-pcen implementation may need adaptation for our pipeline
  (batch dimension handling, GPU compatibility, ONNX export).
- Learnable PCEN adds 4 parameters per mel bin (4 × 256 = 1024 params).
  Negligible compute cost but these parameters could overfit to focal
  recording characteristics, negating the domain-invariance benefit.
  The fixed PCEN variant (v10b) controls for this.
- PCEN replaces our normalization but the raw mel spectrogram (before
  normalization) is the same. If the domain shift is in the mel computation
  itself (e.g., different frequency content in soundscapes vs focal), PCEN
  won't help.

## References

- Wang et al. 2017: "Trainable Frontend for Robust and Far-Field Keyword
  Spotting" — original PCEN paper
- pytorch-pcen: https://github.com/daemon/pytorch-pcen
- Pattern analysis: `notes/2026-04-06-experiment-pattern-analysis.md`
- Exp 021 results: `experiments/021-sed-training/results.md`
- v5j baseline: `experiments/019-sed-ablations/config/hgnet_b0_sed_v5j.yaml`
