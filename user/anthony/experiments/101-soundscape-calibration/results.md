# Results: Soundscape Label Calibration

**Date**: 2026-03-23
**Author**: Anthony
**Status**: Phase 2 complete (raw evaluation + threshold sweep)

## Summary

The mean ensemble of 3 fine-tuned models achieves **0.880 macro AUROC** on 42
evaluable species. With CV-tuned threshold of 0.12, the ensemble reaches
**F1=0.756** on the S22 test set (71% precision, 81% recall) — far better than
the naive 0.5 threshold (F1=0.529). BirdNET/Perch linear probes are not
usable even with threshold tuning (best F1=0.164).

## Model Comparison

### AUROC and AP (42 evaluable species, 739 windows)

| Model | AUROC | AP | S22 AUROC | S22 AP |
|-------|-------|-----|-----------|--------|
| **Ensemble (mean)** | **0.880** | **0.422** | **0.893** | **0.527** |
| ProtoCLR | 0.850 | 0.383 | 0.857 | 0.482 |
| EAT | 0.841 | 0.404 | 0.852 | 0.512 |
| ConvNeXt | 0.815 | 0.343 | 0.805 | 0.445 |
| BirdNET probe | 0.708 | 0.315 | — | — |
| Perch probe | 0.653 | 0.219 | — | — |

## Threshold Sweep Results

### CV-selected thresholds evaluated on S22

Thresholds selected by LOSO CV (median of 8 folds), evaluated on held-out S22.

| Model | Strategy | Threshold | Precision | Recall | F1 |
|-------|----------|-----------|-----------|--------|-----|
| **Ensemble** | **Global** | **0.12** | **0.711** | **0.806** | **0.756** |
| Ensemble | Per-taxa | Amp=0.40, Ave=0.42, Mam=0.50 | 0.911 | 0.521 | 0.663 |
| EAT | Global | 0.12 | 0.673 | 0.756 | 0.712 |
| EAT | Per-taxa | Amp=0.47, Ave=0.50, Mam=0.50 | 0.865 | 0.602 | 0.710 |
| ProtoCLR | Global | 0.25 | 0.803 | 0.583 | 0.676 |
| ConvNeXt | Global | 0.08 | 0.667 | 0.661 | 0.664 |
| BirdNET probe | Global | 0.40 | 0.089 | 1.000 | 0.164 |
| Perch probe | Global | 0.50 | 0.066 | 0.486 | 0.116 |

### Best achievable F1 per taxa (ensemble, full 739 windows)

| Taxa | Best F1 | Threshold | Species |
|------|---------|-----------|---------|
| Amphibia | 0.796 | 0.20 | 14 |
| Aves | 0.459 | 0.15 | 23 |
| Mammalia | 0.444 | 0.15 | 4 |
| Reptilia | — | — | 1 (too few windows) |

Amphibia drives the aggregate F1. Aves performance is substantially lower
even at optimal threshold — birds are harder to detect in these soundscapes.

### Fixed threshold comparison (ensemble, all 739 windows)

| Threshold | TP | FP | FN | TN | Precision | Recall | F1 |
|-----------|-----|-----|------|-------|-----------|--------|------|
| 0.12 (CV-selected) | ~1,400 | ~570 | ~340 | — | 0.711 | 0.806 | 0.756 |
| 0.20 | — | — | — | — | 0.791 | 0.674 | 0.728 |
| 0.30 | 1,256 | 203 | 889 | 28,690 | 0.861 | 0.586 | 0.697 |
| 0.50 | 791 | 52 | 1,354 | 28,841 | 0.938 | 0.369 | 0.529 |
| 0.70 | 466 | 8 | 1,679 | 28,885 | 0.983 | 0.217 | 0.356 |

### CV threshold stability

The CV folds select very different thresholds (ensemble global: [0.30, 0.15,
0.05, 0.05, 0.05, 0.30, 0.10, 0.30]). This instability is expected with
small per-fold sizes (24–60 windows) and site-specific species compositions.
The median (0.12) is reasonable but should be treated as approximate.

## Per-Species Breakdown (Ensemble @ 0.5)

### Reliable tier (AUROC > 0.85, 26 species)

| Species | Common name | Taxa | n+ | AUROC | TP | FP | FN | Prec | Rec |
|---------|-------------|------|----|-------|-----|-----|-----|------|-----|
| 43435 | Black Howling Monkey | Mammalia | 12 | 1.000 | 11 | 0 | 1 | 1.00 | 0.92 |
| hyamac1 | Hyacinth Macaw | Aves | 10 | 1.000 | 2 | 0 | 8 | 1.00 | 0.20 |
| chvcon1 | Chestnut-vented Conebill | Aves | 35 | 0.998 | 0 | 0 | 35 | — | 0.00 |
| nacnig1 | Nacunda Nighthawk | Aves | 11 | 0.998 | 0 | 0 | 11 | — | 0.00 |
| 23158 | Pale-legged Weeping Frog | Amphibia | 175 | 0.973 | 94 | 0 | 81 | 1.00 | 0.54 |
| 65380 | Dwarf Tree Frog | Amphibia | 333 | 0.970 | 208 | 2 | 125 | 0.99 | 0.62 |
| 555146 | Chaco Tree Frog | Amphibia | 210 | 0.967 | 145 | 11 | 65 | 0.93 | 0.69 |
| 24279 | Lesser Snouted Tree Frog | Amphibia | 173 | 0.940 | 109 | 29 | 64 | 0.79 | 0.63 |
| 22973 | Whistling Grass Frog | Amphibia | 213 | 0.928 | 64 | 2 | 149 | 0.97 | 0.30 |
| 22967 | Marbled White-lipped Frog | Amphibia | 155 | 0.917 | 42 | 2 | 113 | 0.95 | 0.27 |

(+ 16 more with AUROC 0.85–0.96)

### Usable tier (0.70–0.85, 6 species)

116570 (Caiman), fusfly1, magant1, 326272 (Weeping Frog), strher2, 67252

### Unreliable tier (< 0.70, 10 species)

67107, limpki, trsowl, 65377, and 6 others with very few positive windows.

## Observations

1. **Optimal threshold is ~0.12, not 0.5.** The compressed probability issue
   (8+ species per window) means 0.5 is far too conservative. CV selects 0.12
   for the ensemble, giving F1=0.756 vs 0.529 at 0.5.

2. **Per-taxa thresholds trade F1 for precision.** Amphibia wants ~0.40, Aves
   wants ~0.42–0.50. The per-taxa approach gets 91% precision but loses recall
   (F1=0.663 vs 0.756 global). Useful for high-confidence-only pseudo-labels.

3. **Amphibia dominates performance.** Best per-taxa F1: Amphibia 0.80, Aves 0.46.
   The high aggregate F1 is driven by amphibians (76% of labeled windows).

4. **Linear probes are not rescuable.** BirdNET/Perch probes have <0.17 F1 even
   at optimal thresholds. The frozen embeddings don't discriminate species well
   enough on multi-species soundscapes. End-to-end fine-tuning (Murilo's models)
   is necessary for this domain.

5. **CV threshold selection is unstable.** Per-fold picks range 0.05–0.30. With
   only 8 sites and 24–60 windows per fold, the optimal threshold depends heavily
   on site-specific species composition.

## Decision

**Use ensemble at global threshold ~0.15–0.20 for pseudo-labeling.**
See `docs/pseudo-label-strategies.md` for strategy comparison.

Recommended approach (Strategy A):
- Threshold 0.15–0.20 (balances precision ~0.75–0.80 with recall ~0.65–0.80)
- Weight pseudo-labels by ensemble probability in downstream training
- Restrict to evaluable species (exclude 28 zero-focal + unreliable tier)

## Artifacts

| File | Description |
|------|-------------|
| `species_detailed.csv` | Per-species × per-model: AUROC, AP, TP/FP/FN/TN @ 0.3/0.5/0.7 |
| `threshold_sweep_{model}.csv` | Full threshold sweep per model (global + per-taxa) |
| `threshold_loso_results.json` | CV-selected thresholds + S22 test results |
| `metrics_{protoclr,convnext,eat,ensemble}.csv` | Per-species AUROC/AP |
| `metrics_probe_{birdnet,perch}.csv` | Per-species AUROC/AP for probes |
| `site_metrics_{protoclr,convnext,eat}.csv` | Per-site macro AUROC |
| `loso_cv.csv` | Leave-one-site-out CV fold results |
| `test_results.json` | S22 test set results |
| `all_model_comparison.json` | All models head-to-head |
| `eval_info.json` | Evaluable species list |
| `call_nocall_results.json` | Call/no-call detection metrics |

## Call/No-Call Detection

Binary task: is any evaluable species vocalizing in a 5s window? (666 positive,
73 negative windows — all labeled windows have at least one species including
sonotypes, but 73 have no evaluable species.)

### AUROC (max score per window)

| Model | AUROC | AP |
|-------|-------|-----|
| Ensemble | 0.870 | 0.983 |
| EAT | 0.850 | 0.980 |
| ProtoCLR | 0.843 | 0.977 |
| ConvNeXt | 0.792 | 0.970 |
| BirdNET probe | 0.694 | 0.958 |
| Perch probe | 0.309 | 0.873 |

### Ensemble confusion at thresholds

| Threshold | TP | FP | FN | TN | Precision | Recall | F1 |
|-----------|-----|-----|-----|-----|-----------|--------|-----|
| 0.10 | 642 | 50 | 24 | 23 | 0.928 | 0.964 | 0.946 |
| 0.15 | 622 | 33 | 44 | 40 | 0.950 | 0.934 | 0.942 |
| 0.20 | 592 | 28 | 74 | 45 | 0.955 | 0.889 | 0.921 |
| 0.50 | 408 | 6 | 258 | 67 | 0.986 | 0.613 | 0.756 |

### Best call score aggregation (ensemble)

| Aggregation | AUROC | AP |
|-------------|-------|-----|
| top5_sum | **0.901** | **0.989** |
| n_above_0.1 | 0.873 | 0.980 |
| max | 0.870 | 0.983 |
| mean | 0.842 | 0.981 |

### Call/no-call observations

- **Probes are always-on**: max score >0.5 for every window, cannot distinguish
  call from no-call. Confirms they are poorly calibrated for soundscapes.
- **Top-5 sum is the best call indicator** (AUROC 0.901) — a real call window
  has multiple species with elevated predictions.
- **Only 73 negative windows** makes this a weak test. These are windows where
  no evaluable species was annotated (may still contain sonotype insects).

### Acoustic indices for call/no-call

See `docs/call-nocall-detector.md` for full details.

**Spectral contrast alone (0.888 AUROC)** nearly matches the ensemble top-5
sum (0.901) for call detection — a single acoustic feature with no ML.

All attempts to improve on raw spectral contrast failed:
- Combined classifiers (LR, ridge, decision trees, GBDT): 0.35–0.53 AUROC
- Isotonic calibration of spectral contrast: 0.472 AUROC (destroyed ranking)

Root cause: only 53 no-call windows across 2 sites. Not enough negatives to
train or calibrate anything. Raw spectral contrast works because it's a
physical measurement that doesn't need calibration data.

**Recommendation**: use raw spectral contrast threshold as a call/no-call gate
before species classification. No calibration.

## Next Steps

- [ ] Apply ensemble + threshold to 176.5h unlabeled soundscapes
- [ ] Per-species threshold tuning for high-AUROC / low-recall species
- [ ] Consider mining hard negatives from unlabeled data for better call/no-call calibration
