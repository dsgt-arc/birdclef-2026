# Results: Weak Supervision Label Fusion (Phase 2)

**Date Completed**: 2026-03-03
**Author**: Anthony

## Summary

3-tier triplet fusion produces usable per-token soft labels for 939K focal and 564K
soundscape tokens across 206 species. Focal labels are well-calibrated with low LF
correlation; soundscape labels are weaker due to domain shift and high LF correlation.

**Verdict**: confirmed (focal), inconclusive (soundscape)

## Observations

### Quantitative

**LF Accuracy (triplet-estimated, mean across 206 species)**

| LF | Focal | Soundscape |
|----|-------|------------|
| BirdNET probe | 0.788 +/- 0.108 | 0.735 +/- 0.191 |
| Perch probe | 0.779 +/- 0.108 | 0.695 +/- 0.168 |
| Indices probe | 0.687 +/- 0.097 | 0.686 +/- 0.166 |
| Recording label | 0.932 +/- 0.087 | N/A |

**LF Correlation (accuracy across species)**

| | Focal | Soundscape |
|--|-------|------------|
| BirdNET-Perch | 0.353 | 0.606 |
| BirdNET-Indices | 0.205 | 0.610 |
| Perch-Indices | 0.221 | 0.598 |

**Soft Label Distribution**

| Metric | Focal | Soundscape |
|--------|-------|------------|
| Tokens | 939,116 | 564,108 |
| Near 0 (<0.1) | 99.1% | 98.5% |
| Near 1 (>0.9) | 0.1% | 0.0% |
| Ambiguous (0.3-0.7) | 0.2% | 0.4% |
| Bimodality | PASS | PASS |
| Tokens with >=1 active species | 88.7% | 82.4% |
| Mean active species/token | 1.8 | 3.2 |
| Species with signal (acc>0.52) | 201/206 | 142/206 |

**Focal Calibration (vs recording-level ground truth)**

| Metric | Value |
|--------|-------|
| Primary species soft label > 0.5 | 53.4% |
| Mean soft label on primary tokens | 0.289 |
| Mean soft label on non-primary tokens | 0.003 |

### Qualitative

- Focal labels are strongly bimodal — the fusion makes confident decisions rather than hedging
- 201/206 species have useful signal on focal; only 5 species stuck at coin-flip accuracy
- Soundscape LF correlations (~0.60) violate the conditional independence assumption more severely than focal (~0.20-0.39)
- 64/206 species have no signal on soundscape (acc = 0.5) — expected domain shift from probes trained on focal data
- Soundscape labels have higher mean active species per token (3.2 vs 1.8), likely inflated by correlated LFs agreeing spuriously

## Analysis

**Focal labels are usable for downstream training.** The 53% primary-species recall at
threshold 0.5 is a soft lower bound — many "misses" are genuinely silent tokens where
the bird is not vocalizing. The 100x separation between primary (0.289) and non-primary
(0.003) mean soft labels indicates strong discriminative signal. For soft-label BCE
training, even modest positive values provide useful gradient.

**Soundscape labels are weaker but additive.** The high LF correlation (0.6) means the
triplet method may over-count agreement, inflating accuracy estimates. For 142/206
species the labels add real signal; for the remaining 64, they are noise. Downstream
consumers should either (a) weight soundscape labels lower than focal, or (b) mask
species with estimated accuracy <= 0.52.

**Indices probe served its purpose.** Despite low standalone AUROC (0.432), the indices
probe provides the critical 3rd LF needed for the triplet method on soundscape. Its
estimated accuracy (0.687) is above the useful threshold. Without it, soundscape fusion
would be impossible (2-LF triplet is degenerate).

### Confounders

- **Domain shift**: Probes trained on focal data applied to soundscape. Feature distributions
  differ substantially (focal NDSI mean 0.21 vs soundscape 0.62). Z-score normalization
  partially mitigates but doesn't eliminate this.
- **Conditional independence violation**: Soundscape LF correlations (~0.6) mean the triplet
  method's accuracy estimates may be biased upward. The true LF accuracies on soundscape
  could be lower than reported.
- **No soundscape ground truth**: Cannot directly evaluate soundscape label quality. The
  sanity checks pass (no inflated species, bimodal distribution), but true calibration is
  unknown.
- **53% focal primary recall**: Could indicate under-confidence on vocalizing tokens, or
  could correctly reflect that birds don't vocalize on every 1s window. Need per-token
  manual inspection to distinguish.

## Artifacts

| Type | Location |
|------|----------|
| Fused soft labels (focal) | `~/scratch/birdclef-2026/outputs/exp013/fused/focal/soft_labels.parquet` |
| Fused soft labels (soundscape) | `~/scratch/birdclef-2026/outputs/exp013/fused/soundscape/soft_labels.parquet` |
| LF accuracies (focal) | `~/scratch/birdclef-2026/outputs/exp013/fused/focal/lf_accuracies.parquet` |
| LF accuracies (soundscape) | `~/scratch/birdclef-2026/outputs/exp013/fused/soundscape/lf_accuracies.parquet` |
| Eval reports | `~/scratch/birdclef-2026/outputs/exp013/eval/{focal,soundscape}/report.txt` |
| Indices probe | `~/scratch/birdclef-2026/outputs/exp013/probes/indices/species_probe.npz` |
| Probe job log | `logs/probe-indices-4590291.out` |
| Fuse job logs | `logs/fuse-4590292_0.out` (focal), `logs/fuse-4590292_1.out` (soundscape) |
| Evaluate job log | `logs/evaluate-4590293.out` |

## Next Steps

- [ ] Feed focal soft labels into exp 012 (EfficientNet finetune) as training targets
- [ ] Feed focal + soundscape soft labels into exp 009 (pretrain) as reconstruction targets
- [ ] For soundscape, filter to 142 species with acc > 0.52 or apply lower sample weight
- [ ] Consider retraining indices probe on combined focal+soundscape features to reduce domain shift
- [ ] Manual inspection of focal tokens where primary species gets < 0.1 — silent windows or fusion errors?

**Leads to**: exp 009 (SSL pretrain with soft targets), exp 012 (EfficientNet finetune with soft labels)
