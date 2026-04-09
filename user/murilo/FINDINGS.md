# Findings

Living document tracking cross-cutting insights from all BirdCLEF 2026
experiments (006-023, 40+ model variants). Replaces
`notes/2026-04-06-experiment-pattern-analysis.md` as the canonical reference.

Last updated: 2026-04-08

---

## Score Progression

Every LB submission in chronological order of improvement:

| Score | Model | Key Change | Exp |
|-------|-------|-----------|-----|
| 0.531 | protoclr-2025 | 2025 model on 2026 data (41/234 overlap) | 009 |
| 0.695-0.736 | exp 011 fulldata | Focal-only, 206 classes, no soundscapes | 011 |
| 0.720-0.767 | exp 012 soundscape | + labeled soundscapes, 234 classes | 012 |
| 0.762 | student-v1 | + pseudo-labels (regressed vs teacher) | 015 |
| 0.795 | SED v1 | SED architecture | 016 |
| 0.827 | SED v1 +pp2 | + TTA + smoothing + priors | 016 |
| 0.830 | v4a +pp2 | + high-res mel (4096 n_fft, 224 mels) | 017 |
| 0.859 | v4b | + AmplitudeToDB normalization | 017 |
| 0.892 | v5g | + HGNetV2-B0 backbone | 019 |
| 0.906 | **v5j** | **+ NNAudio AmplitudeToDB** | 019 |
| 0.909 | v5j-pp2 | + site/hour priors | 021 |
| 0.913 | v5j-4fold | + 4-fold ensemble | 019 |
| 0.917 | v5j-4fold-pp2 | + priors + 4-fold (best SED) | 021 |
| 0.925 | Perch V18 TF | Frozen Perch + ProtoSSM + MLP (best individual) | 020 |
| 0.927 | Perch+SED w=0.5 | Score-level ensemble (ProtoSSM+MLP) | 023 |
| 0.929 | **Perch+SED MLP-only** | **Drop ProtoSSM, MLP probes only (current best)** | 023 |

Target: 0.939 (competition #1). Gap: 0.010.

---

## What Moves the LB Needle

Every positive LB delta involved changing the **input representation**,
**post-processing / inference strategy**, or **model combination**. No
training-time recipe change has ever improved LB.

| Category | Change | Delta | Exp |
|----------|--------|-------|-----|
| Input repr | HGNetV2-B0 backbone (vs EffNet-B0) | **+0.037** | 019 |
| Input repr | AmplitudeToDB normalization (on EffNet) | **+0.034** | 017 |
| Post-proc | TTA + smoothing + TopN | **+0.032** | 016 |
| Architecture | SED (vs clip-level) | **+0.028** | 016 |
| Input repr | AmplitudeToDB + NNAudio (on HGNet) | **+0.014** | 019 |
| Ensemble | 4-fold SED ensemble | **+0.007** | 019 |
| Post-proc | PP gain in ensemble context | **+0.005** | 023 |
| Input repr | Higher mel resolution | **+0.003** | 017 |
| Post-proc | Site/hour priors | **+0.003** | 021 |
| Ensemble | Perch+SED score fusion (MLP-only) | **+0.004** | 023 |
| Ensemble | MLP-only vs ProtoSSM+MLP in ensemble | **+0.002** | 023 |

**Common thread**: change *what the model sees* or *how predictions are
combined*. Never *how the model trains*.

---

## What Never Worked

Every training-time recipe change from the v5j baseline regressed or was flat
on LB, even when it improved offline val/auroc.

| Change | LB Delta | Times Tested | Exps |
|--------|----------|-------------|------|
| CE loss (any variant) | -0.015 to -0.030 | 6x | 016, 017, 021 |
| PCEN normalization | **-0.026** | 1x | 022 |
| Longer window (10s single) | -0.026 | 1x | 019 |
| Pseudo-labels (student) | -0.005 to -0.024 | 2x | 015 |
| 3-channel input | -0.020 | 1x | 019 |
| StochasticDepth | -0.019 | 1x | 017 |
| Waveform MixUp | -0.006 to -0.019 | 2x | 017, 021 |
| AttHead (any variant) | -0.007 to -0.018 | 7x | 021 |
| Backbone scaling (HGNet-B2) | -0.017 | 1x | 021 |
| Waveform augs (noise, stretch) | -0.013 | 1x | 021 |
| Multi-context (any approach) | -0.010 to -0.075 | 5x | 021 |
| Manifold Mixup (Mix2) | -0.007 | 1x | 021 |
| Fulldata training (no val) | -0.006 | 1x | 017 |
| ASL loss | -0.051 | 1x | 017 |
| ProtoSSM in ensemble (ONNX) | -0.002 | 1x | 023 |
| More epochs (25 vs 20) | ~flat | 1x | 021 |

---

## Key Insights

### 1. Only input representation and inference-time changes improve LB

Every positive LB delta involved changing the input representation (mel
resolution, normalization, backbone architecture) or post-processing (TTA,
priors, ensembling). 15+ training recipe changes (loss, augmentation, head,
regularization) have 0 successes. This is the single most striking pattern.

### 2. The offline-LB disconnect is systematic

Models that improve offline metrics via training recipe changes (waveform
MixUp +0.003 val, multi-context +0.008 val, StochasticDepth +0.003 val)
consistently regress on LB. These changes help the model memorize focal
recording patterns better but hurt soundscape generalization.

The domain shift between focal recordings (clean, single-species) and Pantanal
soundscapes (noisy, multi-species, dense choruses) is the fundamental
bottleneck. Our validation set samples from focal data, so offline metrics
reward the opposite of what LB needs.

### 3. The v5j recipe sits at a generalization cliff

Every perturbation to the training recipe falls off the cliff. v5j is not a
perfect recipe -- it found an accidental balance that generalizes from focal
to soundscape. Perturbing that balance in any direction hurts.

### 4. Big leaps came from spectral representation

- HGNetV2-B0 backbone: +0.037 (single biggest gain)
- AmplitudeToDB normalization: +0.014 to +0.034 (backbone-dependent)
- Higher mel resolution: +0.003

All about giving the backbone better input features.

### 5. Perch succeeds because it does not fine-tune

Perch achieves 0.925 with frozen embeddings + lightweight probes + heavy PP.
It succeeds precisely because it does not fine-tune on focal recordings --
the pretrained model generalizes across acoustic domains. Our SED pipeline
fine-tunes on focal data and pays the domain-shift penalty.

### 6. Post-processing is the second biggest lever

- TTA + smoothing + TopN: +0.032 (exp 016)
- PP in ensemble context: +0.005 (exp 023)
- 4-fold ensemble: +0.007 (exp 019)
- Site/hour priors: +0.003 (exp 021)

All inference-time changes that do not touch the model's learned
representations.

### 7. Ensemble of orthogonal models is the third lever

Perch (frozen, domain-general) + SED (fine-tuned, species-specific) are
complementary. The ensemble (0.929) beats the best individual (0.925) by
+0.004 despite using the weaker ONNX Perch backend (0.920 standalone).
The weight curve is flat (0.926-0.927 across w=0.3/0.5/0.7), confirming
both models contribute roughly equally.

### 8. Simpler probes outperform complex temporal models in ensemble

Dropping ProtoSSM and using MLP probes only improved LB from 0.927 to 0.929
(+0.002). ProtoSSM was trained on TF embeddings but runs on ONNX embeddings
at inference, creating a distribution shift. Simpler MLP probes are more
robust to this mismatch. This was the single biggest finding of exp 023.

### 9. PCEN normalization fails despite theoretical motivation

PCEN (exp 022) regressed -0.026, the largest single-variable regression
tested. The pattern analysis on 2026-04-06 specifically identified PCEN as
"high priority" -- it was wrong. The difference from successful input
representation changes: mel resolution and AmplitudeToDB **add information**
(more frequency bins, wider dynamic range). PCEN **transforms** the
representation -- the model had already learned to handle AmplitudeToDB
normalization, and PCEN disrupted that learned mapping.

---

## Remaining Gap Analysis

Current best: 0.929 (MLP-only ensemble). Target: 0.939 (#1). Gap: 0.010.

| Direction | Rationale | Expected Gain | Risk |
|-----------|-----------|---------------|------|
| TF Perch backend | ONNX costs -0.005 standalone; ensemble would recover some | +0.002-0.003 | Kaggle kernel may lack TF support |
| Better Perch probes (OOF, calibration) | MLP probes are simple; OOF calibration could help | +0.001-0.003 | Unknown |
| SED backbone diversity | Train v5j on different backbone for ensemble diversity | +0.001-0.003 | Backbone swap historically high-yield |
| More sophisticated PP fusion | Weight PP per-species or per-site rather than globally | +0.001-0.002 | Complex to tune |
| Additional data sources | Xeno-canto, iNaturalist sound -- more species coverage | +0.001-0.005 | Data cleaning overhead |
| Domain-adapted training | Train on soundscape-augmented focal data | +0.001-0.005 | Pseudo-labels failed before (exp 015) |

Note: individual directions unlikely to close the full 0.010 gap. Need to
stack 3-4 small gains.

---

## Deprioritized Directions

Confirmed dead ends -- do not revisit without strong new evidence:

- **Training recipe changes** (loss, augmentation, head, regularization):
  15+ attempts, 0 successes. The evidence is overwhelming.
- **PCEN normalization**: -0.026 LB, theoretically motivated but empirically
  rejected. Transforms representation rather than adding information.
- **CE loss**: tested 6 times, always regresses. BCE is correct for
  multi-label soundscape prediction.
- **AttHead**: tested 7 times across multiple backbones, always regresses.
  LSEHead is the right choice for this pipeline.
- **Multi-context / longer windows**: tested 5 times, always regresses.
  The 5s window with TTA is the right operating point.
- **ProtoSSM in ONNX ensemble**: simpler MLP probes are strictly better
  when running on ONNX embeddings.
- **Backbone scaling (B0 to B2)**: regressed -0.017 (confounded with AttHead,
  but not worth retesting given diminishing returns).
- **Pseudo-labels / student training**: failed twice in exp 015. Would need
  fundamentally different approach to succeed.
