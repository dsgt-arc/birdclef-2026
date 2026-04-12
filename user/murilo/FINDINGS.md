# Findings

Living document tracking cross-cutting insights from all BirdCLEF 2026
experiments (006-024, 40+ model variants). Replaces
`notes/2026-04-06-experiment-pattern-analysis.md` as the canonical reference.

Last updated: 2026-04-09

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

Target: 0.948 (competition #1). Gap: 0.019.

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
| CE loss (any variant) | -0.015 to **-0.409** | 7x | 016, 017, 021, **027** |
| PCEN normalization | **-0.026** | 1x | 022 |
| Longer window (10s single) | -0.026 | 1x | 019 |
| Pseudo-labels (student) | -0.005 to -0.024 | 2x | 015 |
| 3-channel input | -0.020 | 1x | 019 |
| StochasticDepth | -0.019 | 1x | 017 |
| Waveform MixUp (any variant) | -0.019 to **-0.031** | 3x | 017, 021, **027** |
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

### 10. The LB gap is a data coverage problem, not model quality (exp 024)

The v5j model achieves **0.9958 macro-AUC** on the 75 species present in labeled
soundscapes — near-perfect. Yet it scores only 0.906 on the public LB (234 species).
The gap is almost entirely explained by **159 species that never appear in labeled
soundscape data**. These species are trained only on focal recordings and must
generalize blindly to the soundscape domain. No amount of training recipe tuning
can fix this — the model needs soundscape-domain data for these 159 species.

Full analysis: `notes/2026-04-09-inference-eda-deep-dive.md`

### 11. Max-across-folds dramatically outperforms averaging for rare species (exp 024)

For pseudo-labeling, `max(fold0, fold1, fold2, fold3)` per species per segment
recovers **+29 unseen species** at >0.5 threshold compared to single-fold (80 → 109/159).
4-fold averaging actually **hurts** coverage (-12 species vs single-fold) because it
smooths out confident individual-fold predictions for rare species. Each fold contributes
2-6 unique species that no other fold detects.

| Strategy | Coverage @>0.5 | Zero Coverage |
|----------|---------------|---------------|
| Single-fold | 80 / 159 | 79 |
| Avg-4fold | 68 / 159 | 91 (worse!) |
| **Max-4fold** | **109 / 159** | **50** |

34 species remain invisible even with max-folds at >0.3 — these need Perch.

### 12. 12.6% of focal "confusions" are correct background species detections (exp 024)

9,498 / 75,553 wrong top-1 predictions on focal recordings match a secondary species
label. The model is detecting real background species that happen to be louder than
the primary species in that 5s chunk. This means the model's multi-label capability
is better than focal metrics suggest, and its predictions on multi-species soundscapes
may be more trustworthy than the 0.716 top-1 accuracy implies.

### 13. Domain shift is species-specific, not global (exp 024)

The overall logit distribution is similar between focal and soundscape recordings
(mean logit -9.20 vs -9.23). But which species get high logits changes dramatically:
- 75 seen species: mean max-prob 0.579 on soundscapes (actively detected)
- 159 unseen species: mean max-prob 0.036 (confidently predicted as absent)

The model isn't generally confused by soundscapes — it's specifically blind to species
it hasn't seen in the soundscape domain.

### 14. Focal sample count moderately predicts soundscape coverage (exp 024)

Spearman r=0.459 between focal recording count and pseudo-label coverage on soundscapes.
Species with 200+ focal recordings: 75% get pseudo-labels at >0.5. Species with <10:
only 6%. However, **54 species with 200+ focal recordings still get zero soundscape
predictions** — these are pure domain-shift failures, not data scarcity. Manual
inspection needed to determine if they're genuinely absent from Pantanal soundscapes.

---

## Remaining Gap Analysis

Current best: 0.929 (MLP-only ensemble). Target: 0.948 (#1). Gap: 0.019.

**Data-centric directions** (highest priority — see `notes/2026-04-09-data-centric-pivot.md`):

| Direction | Rationale | Expected Gain | Risk |
|-----------|-----------|---------------|------|
| **Pseudo-labeling v2** | 127K unlabeled soundscape segments, 0.929 teacher, addresses 159 unseen species | **+0.005-0.010** | Medium — exp 015 failed, but teacher is 5× stronger + SED architecture now |
| **Backbone diversity** | Nikita used 7 models × 5 backbones; we use 1. Diverse ensemble + diverse teachers for PL iteration | +0.003-0.007 | Low — proven technique |
| **VAD cleaning** | Remove speech contamination from focal data, improve data quality | +0.001-0.002 | Low — data hygiene |
| **Hard negative mining** | Use unlabeled data to find confident FPs, add as explicit negatives | +0.001-0.003 | Medium — needs careful design |

**Inference/ensemble directions** (lower priority):

| Direction | Rationale | Expected Gain | Risk |
|-----------|-----------|---------------|------|
| Overlap reconstruction | Nikita's 20s windows + delta blending (inference-only) | +0.001-0.003 | Low |
| SED backbone diversity | Train v5j on different backbone for ensemble diversity | +0.001-0.003 | Low |
| Better Perch probes (OOF, calibration) | MLP probes are simple; OOF calibration could help | +0.001-0.003 | Unknown |
| More sophisticated PP fusion | Weight PP per-species or per-site rather than globally | +0.001-0.002 | Complex to tune |

Note: the gap widened from 0.017 to 0.019 (#1 moved from 0.946 to 0.948). The #1 team
is almost certainly using unlabeled data (Boredom says ceiling without unlabeled is ~0.94-0.945).
Pseudo-labeling is the single highest-value bet for closing the gap.

---

## Deprioritized Directions

Confirmed dead ends -- do not revisit without strong new evidence:

- **Training recipe changes** (loss, augmentation, head, regularization):
  15+ attempts, 0 successes. The evidence is overwhelming.
- **PCEN normalization**: -0.026 LB, theoretically motivated but empirically
  rejected. Transforms representation rather than adding information.
- **CE loss**: tested 7 times (incl. exp 027 v10a=0.494), always regresses.
  Root cause (exp 027): CE optimizes relative ranking (softmax), producing
  logits ~0 magnitude. Sigmoid maps all species to ~0.50 → random on soundscapes.
  BCE pushes absent species to logit=-9, robust to domain shift. Competitors
  succeed with CE via ensembles (min-reduction) or pseudo-labels, not standalone.
- **AttHead**: tested 7 times across multiple backbones, always regresses.
  LSEHead is the right choice for this pipeline.
- **Multi-context / longer windows**: tested 5 times, always regresses.
  The 5s window with TTA is the right operating point.
- **ProtoSSM in ONNX ensemble**: simpler MLP probes are strictly better
  when running on ONNX embeddings.
- **Backbone scaling (B0 to B2)**: regressed -0.017 (confounded with AttHead,
  but not worth retesting given diminishing returns).
- **Pseudo-labels / student training (exp 015 approach)**: failed twice with
  0.767 teacher + clip-level architecture. The approach itself is NOT dead —
  it needs fundamentally better conditions: stronger teacher (0.929), SED
  architecture, power transform filtering. See exp 024/025.
