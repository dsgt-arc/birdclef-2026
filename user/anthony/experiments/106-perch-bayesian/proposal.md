# Proposal: Perch v2 Bayesian Label Transfer

**Status**: proposed
**Created**: 2026-04-06
**Author**: Anthony

## Hypothesis

Perch v2's pretrained label output (14,795-d bird taxonomy logits) contains
correlational structure that can be exploited via Bayesian inference to transfer
species predictions to BirdCLEF 2026's 234 targets — including non-bird taxa —
without gradient-based training. By learning a probabilistic mapping on 739
labeled soundscape windows and propagating it to 10,592 unlabeled soundscapes,
we can produce calibrated pseudo-labels that outperform the direct linear probe
baseline (exp 105: 0.684 cmAP focal, 0.052 sc_cmap).

## Background

Exp 105 showed that frozen Perch v2 embeddings (1536-d) achieve strong focal
cmAP (0.684) but poor soundscape cmAP (0.052), indicating severe domain shift.
The Perch `label` output (14,795 logits over its pretrained taxonomy) is an
untapped signal: even for non-bird species, correlated activations across Perch's
taxonomy may encode useful acoustic signatures (e.g., a frog call might
consistently co-activate certain bird species logits).

Key observations motivating this approach:
- **739 labeled windows** is too few for discriminative training but enough for
  Bayesian posterior estimation over a structured label space
- Perch's 14,795 labels likely have sparse but informative correlations with
  BirdCLEF targets (same species under different names, acoustic similarity)
- Frame-level analysis (Perch processes 5s windows, soundscapes are 60s with
  12 windows) allows temporal aggregation tuning — some species need wider
  context windows than others
- Unlabeled soundscapes (176.5h) can participate as evidence in a semi-supervised
  Bayesian framework (e.g., empirical Bayes on the marginal label distribution)

Related: exp 101 (soundscape calibration), exp 105 (Perch frozen features).

## Method

### Approach

Two-phase experiment:

**Phase 1 — Feature extraction**: Run Perch v2 ONNX on all soundscape files
(labeled + unlabeled) and all focal recordings, capturing `embedding` (1536-d)
and `label` (14,795-d) outputs. Store as sharded parquet for easy analysis.

**Phase 2 — Bayesian analysis**: On the labeled subset, estimate the posterior
P(BirdCLEF species | Perch label activations) using the 739 ground-truth windows.
Explore:
- Sparse correlation mapping (which Perch logits predict which BirdCLEF targets)
- Per-species activation profiles (mean/variance of Perch logits when species is present)
- Temporal aggregation: how frame-level predictions should be pooled across a
  60s soundscape (species-specific window widths)
- Semi-supervised extension: use unlabeled soundscape statistics as empirical prior

### Setup

| Component | Details |
|-----------|---------|
| Data | All 10,658 soundscapes (labeled + unlabeled) + 35,549 focal recordings |
| Compute | CPU-only extraction (cpu-small, ~6h for focal sharded 8-way); analysis is lightweight |
| Dependencies | onnxruntime, numpy, pandas, pyarrow, scipy, scikit-learn; reuse exp 105's `perch_onnx.py` |
| Code | Extraction adapted from exp 105; analysis scripts new |

### Procedure

1. Extract Perch `embedding` + `label` for all soundscape files (labeled + unlabeled), save as parquet
2. Extract Perch `embedding` + `label` for all focal recordings, save as sharded parquet
3. Exploratory analysis: correlation structure between Perch logits and BirdCLEF ground truth
4. Build Bayesian mapping: P(target | perch_logits) using labeled windows
5. Evaluate on held-out labeled windows (reuse exp 102/105 split, seed=42)
6. If promising, generate pseudo-labels for all unlabeled soundscapes
7. Tune temporal aggregation (per-species optimal window width over 5s frames)

### Variables

- **Independent**: Perch label activations (14,795-d), temporal aggregation width, prior structure
- **Dependent**: Species prediction accuracy on held-out labeled windows (cmAP, AUROC)
- **Controlled**: Perch v2 model (frozen ONNX), BirdCLEF 2026 taxonomy (234 species), split seed (42)

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| sc_cmap | Macro-averaged cmAP on soundscape validation windows (primary) |
| AUROC | Per-species and macro AUROC on labeled windows |
| Correlation sparsity | Number of Perch logits significantly correlated with each BirdCLEF target |
| Coverage | Fraction of 234 targets with at least one informative Perch logit correlate |

### Baseline

| Model | Focal cmAP | Soundscape cmAP | Source |
|-------|-----------|-----------------|--------|
| Perch linear probe | 0.684 | 0.052 | Exp 105 phase 1 |
| Perch SED head | 0.695 | 0.072 | Exp 105 phase 2 |
| Teacher ensemble | — | 0.880 AUROC | Exp 101 |

### Success Criteria

- **Confirm if**: Bayesian label transfer achieves sc_cmap > 0.10 (beats exp 105's 0.072) OR reveals interpretable correlation structure useful for downstream models
- **Reject if**: Perch logits show no meaningful correlation with non-bird BirdCLEF targets AND sc_cmap <= 0.05

## Limitations

- Perch is bird-specialized; its 14,795 taxonomy may have zero informative logits for amphibians, mammals, and insects (which dominate BirdCLEF 2026 soundscapes)
- 739 labeled windows is small for estimating correlations in 14,795-d space — heavy regularization or dimensionality reduction needed
- Bayesian approach assumes Perch logit activations are informative and approximately stationary across labeled and unlabeled soundscapes
- No gradient-based fine-tuning means we're limited by what the frozen model already encodes
