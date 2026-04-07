# Experiment Pattern Analysis: What Works vs What Doesn't

*2026-04-06*

Systematic analysis across all BirdCLEF 2026 experiments (006–021, 40+ model
variants) to identify patterns in what improved LB vs what regressed. Motivated
by exp 021 Phase G exhausting single-variable training recipe changes without
breaking past v5j's 0.906 LB.

## Sources

- `user/murilo/MODEL-RUNS.md` — single source of truth for all model training
- `user/murilo/experiments/021-sed-training/results.md` — Phase A–G results
- `user/murilo/experiments/019-sed-ablations/` — v5a–v5n ablations
- `user/murilo/experiments/017-mel-overhaul/` — v4a–v4k ablations
- `user/murilo/experiments/016-pp2/` — SED v1–v3 + post-processing
- `user/murilo/experiments/020-perch-onnx/` — Perch ProtoSSM pipeline (0.925 LB)

---

## What Actually Moved the LB Needle

Every positive LB delta involved changing the **input representation** or
**post-processing / inference strategy**. No training-time recipe change has
ever improved LB.

| Change | From → To | LB Delta | Experiment |
|--------|-----------|----------|------------|
| SED architecture (vs clip-level) | 0.767 → 0.795 | **+0.028** | exp 016 |
| Post-processing (TTA + smoothing + TopN) | 0.795 → 0.827 | **+0.032** | exp 016 |
| Mel resolution (n_fft 1024→4096, 128→224 mels) | 0.827 → 0.830 | **+0.003** | exp 017 |
| AmplitudeToDB normalization (on EffNet) | 0.825 → 0.859 | **+0.034** | exp 017 v4b |
| HGNetV2-B0 backbone (vs EffNet-B0) | 0.855 → 0.892 | **+0.037** | exp 019 v5g |
| AmplitudeToDB + NNAudio (on HGNet) | 0.892 → 0.906 | **+0.014** | exp 019 v5j |
| 4-fold ensemble | 0.906 → 0.913 | **+0.007** | exp 019 |
| Site/hour priors | 0.906 → 0.909 | **+0.003** | exp 021 Phase A |

**Common thread**: these changes alter *what the model sees* (spectral
representation, backbone architecture, normalization) or *how predictions are
combined* (ensembling, priors, TTA) — never *how the model trains*.

## What Never Worked

Every training-time recipe change from the v5j baseline regressed or was flat
on LB, even when it improved offline val/auroc.

| Change | LB Delta | Times Tested | Experiments |
|--------|----------|-------------|-------------|
| CE loss (any variant) | −0.015 to −0.030 | 6× | exp 016 v2/v3c, exp 017 v4i/v4k, exp 021 v6a/v6f/v6g |
| Waveform MixUp | −0.006 to −0.019 | 2× | exp 017 v4j, exp 021 v6b |
| Waveform augs (noise, stretch) | −0.013 | 1× | exp 021 v6c |
| AttHead (any variant, any backbone) | −0.007 to −0.018 | 7× | exp 021 v6e–v6k |
| Multi-context (any approach) | −0.010 to −0.075 | 5× | exp 021 v7a–v7c, v8a, v8b |
| Manifold Mixup (Mix2) | −0.007 | 1× | exp 021 v6l |
| Longer window (10s single) | −0.026 | 1× | exp 019 v5i |
| More epochs (25 vs 20) | ~flat | 1× | exp 021 v6d |
| 3-channel input | −0.020 | 1× | exp 019 v5k |
| Fulldata training (no val) | −0.006 | 1× | exp 017 v4b-fulldata |
| Pseudo-labels (student training) | −0.005 to −0.024 | 2× | exp 015 |
| Backbone scaling (HGNet-B2) | −0.017 | 1× | exp 021 v6h |
| StochasticDepth | −0.019 | 1× | exp 017 v4e |
| ASL loss | −0.051 | 1× | exp 017 v4d |

---

## Key Insights

### 1. The only gains came from changing what the model sees, not how it learns

Every positive LB delta involved changing the **input representation** (mel
resolution, normalization, backbone architecture) or **post-processing**
(TTA, priors, ensembling). No training-time recipe change (loss function,
augmentation strategy, head architecture, regularization) has ever improved LB.
This is the single most striking pattern across 40+ experiments.

### 2. The offline-LB disconnect is systematic, not random

Models that improve offline metrics via training recipe changes (waveform
MixUp +0.003, multi-context +0.008, StochasticDepth +0.003, manifold mixup
±0.000) consistently regress on LB. This suggests these changes help the model
**memorize focal recording patterns** better (higher val/auroc on focal data)
but **hurt soundscape generalization**.

The domain shift between focal recordings (clean, single-species, variable
quality) and Pantanal soundscapes (noisy, multi-species, dense choruses) is
the fundamental bottleneck. Our validation set is sampled from focal data,
so offline metrics reward focal-recording performance — the opposite of what
LB needs.

### 3. The v5j recipe sits at a generalization cliff

Every perturbation to the training recipe falls off the cliff. This isn't
because v5j is a perfect recipe — it's because our training data and evaluation
data are from fundamentally different domains. Changes that make the model better
at classifying focal recordings make it worse on soundscapes. The recipe found
an accidental balance that generalizes; perturbing that balance in any direction
hurts.

### 4. The big leaps came from better spectral representation

- HGNetV2-B0 backbone: **+0.037 LB** (the single biggest gain across all experiments)
- AmplitudeToDB normalization: **+0.014 to +0.034 LB** (backbone-dependent)
- Higher mel resolution: **+0.003 LB**

These are all about giving the backbone **better input features to work with**.
The backbone itself matters enormously (HGNetV2 >> EffNet-B0), and how you
preprocess the mel spectrogram matters even more (AmplitudeToDB was +0.034).

### 5. The Perch pipeline (0.925) succeeds for the opposite reason

Perch achieves 0.925 LB with frozen embeddings + lightweight probes + heavy
post-processing. It succeeds precisely because it **doesn't fine-tune on focal
recordings** — the pretrained Perch model generalizes across acoustic domains.
Our SED pipeline fine-tunes on focal data and pays the domain-shift penalty.

### 6. Post-processing has been our second biggest lever

- TTA + smoothing + TopN: +0.032 (exp 016)
- 4-fold ensemble: +0.007 (exp 019)
- Site/hour priors: +0.003 (exp 021)

These are all inference-time changes. They work because they don't touch the
model's learned representations — they just smooth, combine, or condition
predictions.

---

## Implications for Next Steps

The pattern says: **stop changing the training recipe and start changing what
the model sees or how predictions are combined.**

**Note on TTA**: All LB scores from v5g onward already include ±2.5s time-shift
TTA (50/25/25 weighting), temporal smoothing (Gaussian), and TopN filtering —
these are globally enabled in `submission.py`. The 0.906 is NOT a raw score.
This was discovered 2026-04-06 when investigating G.3 (TTA implementation).

### High-value directions

1. **Bridge the domain gap** — the model trains on clean focal recordings
   but must predict on noisy multi-species soundscapes. Options:
   - Train on labeled soundscapes more heavily (currently ~2% of training data)
   - Use soundscape-style augmentation to make focal data sound more like
     soundscapes (background mixing, SNR reduction, multi-label overlay)
   - Fine-tune on unlabeled soundscapes with pseudo-labels (exp 015 failed
     with LoRA models, but SED pipeline might respond differently)
   - Domain adaptation / domain-adversarial training

2. **Input representation improvements** — our biggest lever historically:
   - Higher fmax (currently null = 16kHz; insect stridulation peaks at 8–20kHz)
   - Different mel filter bank (bark scale, constant-Q, gammatone)
   - Multi-resolution input (low-res for birds, high-res for insects)
   - **PCEN normalization** (per-channel energy normalization) — widely used in
     bioacoustics (Google's Perch uses it internally). PCEN adapts to local
     signal energy, suppressing stationary noise while preserving transients.
     Unlike AmplitudeToDB (our current norm, +0.014 to +0.034 LB gain), PCEN
     is designed for variable-SNR environments like outdoor soundscapes —
     directly addresses the domain shift between clean focal recordings and
     noisy Pantanal soundscapes. High priority.

3. **Hybrid SED + Perch approach** — our two best pipelines are complementary:
   - SED (0.906 single, 0.917 ensemble): fine-tuned on focal data, strong on
     species with focal recordings, weak on soundscape-only species
   - Perch (0.925): frozen embeddings, strong on domain generalization, heavy
     post-processing compensates for simpler model
   - Combining SED predictions with Perch features or ensembling SED+Perch
     could capture both strengths. Options: feature concatenation, score-level
     fusion, or using Perch embeddings as auxiliary input to the SED head.

4. **Inference-time improvements** — the only category that consistently works:
   - Better ensemble strategies (model diversity, not just fold diversity)
   - Improved post-processing (Perch pipeline's 6-stage PP is much more
     sophisticated than our simple priors)
   - TTA already implemented (±2.5s shifts) — could test different shift
     amounts or more shifts, but likely diminishing returns

### Deprioritized directions

- **Training recipe changes** (loss functions, augmentation strategies, head
  architectures, regularization) — 14 attempts, 0 successes. The evidence is
  overwhelming.
- **Backbone scaling** — HGNet-B2 with AttHead regressed; but this was
  confounded with the head swap. A clean test with LSEHead could still help,
  though the gain ceiling is likely small given EffNet→HGNet was already +0.037.

---

## Full LB Score Timeline

For reference, here is every LB submission we've made, in chronological order
of improvement:

| Score | Model | Key Change |
|-------|-------|-----------|
| 0.531 | protoclr-2025 (exp 009) | 2025 model on 2026 data (41/234 overlap) |
| 0.695–0.736 | exp 011 fulldata models | Focal-only, 206 classes, no soundscapes |
| 0.720–0.767 | exp 012 soundscape models | + labeled soundscapes, 234 classes |
| 0.762 | exp 015 student-v1 | + pseudo-labels (regressed) |
| 0.795 | exp 016 SED v1 | SED architecture |
| 0.827 | exp 016 SED v1 +pp2 | + TTA + smoothing + priors |
| 0.830 | exp 017 v4a +pp2 | + high-res mel |
| 0.859 | exp 017 v4b | + AmplitudeToDB |
| 0.892 | exp 019 v5g | + HGNetV2-B0 backbone |
| **0.906** | **exp 019 v5j** | **+ NNAudio AmplitudeToDB** |
| 0.909 | exp 021 v5j-pp2 | + site/hour priors |
| 0.913 | exp 019 v5j-4fold | + 4-fold ensemble |
| **0.917** | **exp 021 v5j-4fold-pp2** | **+ priors + 4-fold (best SED)** |
| **0.925** | **exp 020 Perch V18** | **Frozen Perch + ProtoSSM (best overall)** |
