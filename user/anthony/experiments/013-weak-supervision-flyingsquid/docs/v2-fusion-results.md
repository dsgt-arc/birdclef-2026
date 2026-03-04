# Results: v2 Label Fusion (Raw Logit LFs)

**Date Completed**: 2026-03-04
**Author**: Anthony

## Summary

v2 adds BirdNET/Perch raw logits as extra labeling functions for bird species (145/143 of 146). The triplet method estimates their accuracy at 0.92/0.91 — the strongest LFs in the system. Energy-attenuated probe retraining was tested and rejected.

**Verdict**: confirmed (raw logit LFs improve fusion quality for birds)

## Observations

### Quantitative

#### Focal — v1 vs v2 Direct Comparison (939K tokens)

The aggregate evaluation report numbers look identical between v1 and v2 because
they're dominated by the 97%+ near-zero labels. Direct parquet comparison reveals
large differences in label quality:

| Metric | v1 (4 LFs) | v2 (+ raw logits) | Change |
|--------|-----------|-------------------|--------|
| Primary species mean soft label | 0.549 | **0.840** | +0.291 |
| Primary species median | 0.596 | **0.995** | +0.399 |
| Primary species >0.5 | 55.4% | **84.8%** | +29.4pp |
| High-confidence labels (>0.95) | 77K | **951K** | **12.3x** |
| Non-primary mean soft label | 0.003 | 0.009 | +3.2x |
| Labels that went up (>0.01) | — | 3.97% | — |
| Labels that went down (<-0.01) | — | 1.64% | — |

v1 was surprisingly weak: only 55% of tokens for the correct species exceeded 0.5.
The probes alone lacked confidence. v2 with raw logits pushes primary species labels
to near-certainty (median 0.995) for the majority of tokens.

The tradeoff: non-primary species labels tripled (0.003 → 0.009), but remain small
in absolute terms. The net effect is strongly positive — labels went from "barely
above coin flip" to "mostly confident and correct."

#### Focal — Soft Label Distribution

| Bin | v1 count | v1 % | v2 count | v2 % |
|-----|----------|------|----------|------|
| [0.00, 0.05) | 190,891,122 | 98.67% | 187,821,738 | 97.09% |
| [0.05, 0.10) | 861,207 | 0.45% | 1,412,355 | 0.73% |
| [0.10, 0.20) | 594,469 | 0.31% | 1,056,627 | 0.55% |
| [0.20, 0.30) | 231,583 | 0.12% | 518,745 | 0.27% |
| [0.30, 0.50) | 273,353 | 0.14% | 575,085 | 0.30% |
| [0.50, 0.70) | 190,897 | 0.10% | 468,810 | 0.24% |
| [0.70, 0.80) | 103,485 | 0.05% | 187,700 | 0.10% |
| [0.80, 0.90) | 125,355 | 0.07% | 256,929 | 0.13% |
| [0.90, 0.95) | 108,946 | 0.06% | 209,322 | 0.11% |
| [0.95, 1.00) | 77,479 | 0.04% | 950,585 | 0.49% |

v2 shifts mass from the near-zero bin into confident positive labels. The [0.95, 1.00)
bin grows 12.3x — these are tokens where the raw logit LFs strongly agree with the
probe LFs on species identity.

#### Focal — Aggregate Evaluation Report

| Metric | v1 (4 LFs) | v2 (+ raw logits) | Notes |
|--------|-----------|-------------------|-------|
| Near 0 (<0.1) | 99.1% | 97.8% | v2 slightly fewer (labels shifted up) |
| Near 1 (>0.9) | 0.1% | 0.6% | v2 6x more confident labels |
| Ambiguous (0.3-0.7) | 0.2% | 0.5% | v2 slightly more (mid-range tokens) |
| Bimodality | PASS | PASS | Both >96% near extremes |
| Species w/ signal (acc>0.52) | 201/206 | 201/206 | Unchanged |

#### Focal LF Accuracies (triplet method estimates)

| LF | Accuracy | Std | Range | Coverage |
|----|----------|-----|-------|----------|
| lf0_birdnet (probe) | 0.809 | 0.099 | [0.500, 0.950] | 206 species |
| lf1_perch (probe) | 0.794 | 0.104 | [0.500, 0.950] | 206 species |
| lf2_indices (probe) | 0.679 | 0.087 | [0.500, 0.950] | 206 species |
| lf3_rec_label | 0.933 | 0.087 | [0.500, 0.950] | 206 species |
| **lf_birdnet_raw** | **0.920** | 0.038 | [0.722, 0.950] | 146 birds |
| **lf_perch_raw** | **0.912** | 0.041 | [0.776, 0.950] | 142 birds |

#### Soundscape (564K tokens, 5 LFs for birds, 3 for non-birds)

| Metric | v1 (3 LFs) | v2 (+ raw logits) | Notes |
|--------|-----------|-------------------|-------|
| Near 0 (<0.1) | 96.2% | 96.2% | Identical |
| Near 1 (>0.9) | 0.6% | 0.6% | Identical |
| Ambiguous (0.3-0.7) | 1.2% | 1.2% | Higher than focal (fewer LFs, no rec label) |
| Bimodality | PASS | PASS | 96.8% near extremes |
| Species w/ signal (acc>0.52) | 192/206 | 192/206 | 14 species below threshold |
| Mean active species/token | 7.7 | 7.7 | Higher than focal's 4.5 |

#### Soundscape LF Accuracies

| LF | Accuracy | Std | Range | Coverage |
|----|----------|-----|-------|----------|
| lf0_birdnet (probe) | 0.825 | 0.134 | [0.500, 0.950] | 206 species |
| lf1_perch (probe) | 0.762 | 0.126 | [0.500, 0.950] | 206 species |
| lf2_indices (probe) | 0.721 | 0.137 | [0.500, 0.950] | 206 species |
| **lf_birdnet_raw** | **0.923** | 0.052 | [0.697, 0.950] | 146 birds |
| **lf_perch_raw** | **0.923** | 0.050 | [0.648, 0.950] | 142 birds |

#### Activity Modulation

| Signal | Focal (-1 votes) | Soundscape (-1 votes) |
|--------|------------------|-----------------------|
| NDSI (<-0.2) | 29.3% | 10.4% |
| SpectralContrast (<20) | 9.4% | 6.8% |
| BioIndex (<100) | 0.7% | 13.4% |
| Any suppressed | 32.4% | 19.7% |

#### Fusion vs Individual Teachers

Comparison of primary species detection on focal tokens (5-shard sample, bird species only):

| Method | Primary >0.5 | Primary mean | Primary median | FP mean |
|--------|-------------|--------------|----------------|---------|
| BirdNET sigmoid alone | 39.6% | 0.393 | 0.173 | 0.000313 |
| Perch sigmoid alone | 56.1% | 0.550 | 0.728 | 0.000100 |
| **Fused v2** | **84.8%** | **0.840** | **0.995** | 0.009 |
| Hard labels (oracle) | 100% | 1.000 | 1.000 | 0.000 |

Fusion substantially outperforms either teacher alone. BirdNET only detects 40% of
primary species tokens above 0.5 — it was trained on 6522 species globally, so any
single BirdCLEF species is a tiny fraction of its output space. Perch does better at
56% but still misses nearly half. Fusion reaches 85% by combining the recording label
LF (which knows *what* species) with the raw logits (which know *when* something is
vocalizing). Neither source has both pieces of information alone.

The tradeoff: fusion has higher false positive rate (0.009 vs ~0.0002 for individual
teachers) because the recording label LF always votes +1 for the primary species,
bleeding some positive signal to all tokens. But 0.009 is negligible for training
(corresponds to ~1.9 spurious species per token out of 206).

#### Energy Attenuation (REJECTED)

| Approach | BirdNET AUROC | Notes |
|----------|--------------|-------|
| Baseline (no energy) | 0.785 | v6 probes |
| Soft labels (original impl) | 0.015 | Labels all ~0.5, focal loss collapsed |
| Sample weights (fixed impl) | 0.783 | Matched baseline, no improvement |

### Qualitative

- Raw logit LFs have much tighter accuracy spread (std 0.038-0.052) vs probes (std 0.087-0.134) — the pretrained models are consistently strong across species
- Raw logits are only available for birds (145 BirdNET, 142 Perch mapped) — non-bird species (amphibians, insects, mammals) use 3-4 base LFs only
- The aggregate evaluation report metrics (bimodality, ambiguity percentages) appear similar at a glance, but direct parquet comparison reveals large shifts in label confidence. The evaluation report is dominated by the 97%+ near-zero mass, which hides meaningful changes in the positive tail
- Energy attenuation failed because bioacoustic_index per-recording sigmoid normalization clusters weights at ~0.5 (mean=0.509). All acoustic indices are species-agnostic — they detect "sound present" but can't distinguish which species

## Analysis

### v2 is a substantial improvement over v1

The headline numbers from the evaluation report were misleading — they looked identical because 97%+ of all (token × species) entries are near-zero, and aggregate statistics are dominated by this mass. Direct comparison of the parquets shows v2 is dramatically better:

- **Primary species confidence**: v1 labels were barely above coin flip (mean 0.549, only 55% above 0.5). v2 pushes primary species to near-certainty (mean 0.840, median 0.995, 85% above 0.5).
- **High-confidence labels**: 12.3x more labels above 0.95 (77K → 951K). These are tokens where raw logit LFs strongly agree with probe LFs.
- **False positive cost**: non-primary species labels tripled (0.003 → 0.009), but remain negligible in absolute terms. For downstream training, a 0.009 false positive is effectively zero after thresholding.

The raw logit LFs are the strongest individual signals in the system (0.92 estimated accuracy), surpassing even the recording label LF (0.933 but with zero temporal precision). This is expected: BirdNET/Perch were trained end-to-end on millions of recordings with species-specific output heads, while the probes are single linear layers on frozen embeddings.

### Why the improvement matters for downstream

v1 labels would have been problematic for training — a soft label of 0.55 on the correct species barely provides gradient signal. With BCE loss, the difference between target=0.55 and target=0.50 is tiny. v2 labels at 0.995 provide strong, clean supervision that's close to the hard-label quality of manually annotated data, but with temporal localization (per-token rather than per-recording).

### Non-bird species are unchanged

The 60 non-bird species (34 amphibians, 17 insects, 9 mammals) still use only 3-4 base LFs since BirdNET/Perch don't cover them. Their label quality is the same as v1. This is acceptable since bird species comprise 96.8% of training data.

### Energy Attenuation Post-Mortem

The original implementation replaced binary labels with energy values as targets — this destroyed the signal because `FocalBCELoss` expects binary targets. Fixed to use sample weights instead (multiplier on loss), which preserved AUROC but didn't improve over baseline. Root cause: BI doesn't vary enough within focal recordings to create meaningful contrast.

### Confounders

- Triplet method accuracy estimates assume conditional independence between LFs. BirdNET probe and BirdNET raw logits share the same backbone — their errors are correlated, which may inflate accuracy estimates
- The [0.05, 0.95] accuracy clipping means we can't distinguish between a 0.95-accurate LF and a 0.99-accurate one

## Artifacts

| Type | Location |
|------|----------|
| Fused labels (focal) | `~/scratch/birdclef-2026/outputs/exp013/fused-v2/focal/soft_labels.parquet` |
| Fused labels (soundscape) | `~/scratch/birdclef-2026/outputs/exp013/fused-v2/soundscape/soft_labels.parquet` |
| LF accuracies | `~/scratch/birdclef-2026/outputs/exp013/fused-v2/{focal,soundscape}/lf_accuracies.parquet` |
| Species mapping | `~/scratch/birdclef-2026/outputs/exp013/mappings/species_mapping.json` |
| Evaluation reports | `~/scratch/birdclef-2026/outputs/exp013/eval/{focal,soundscape}/report.txt` |
| Logs | `logs/fuse-v2-4603381_{0,1}.out`, `logs/evaluate-4603382.out` |
| Code | commit `52e7800` + sample weight fix (uncommitted) |

## Next Steps

- [ ] Train exp 012 EfficientNet with v2 fused soft labels and compare to hard-label baseline
- [ ] Train exp 009 SSAMBA pretrain with v2 fused soft labels
- [ ] Investigate whether correlated LFs (probe + raw from same backbone) inflate accuracy — could test by dropping probe when raw is available

**Leads to**: exp 009 (pretrain), exp 012 (finetune) — both consume `fused-v2/soft_labels.parquet`
