# Call/No-Call Detector via Acoustic Indices

## Motivation

The species-level ensemble (0.880 AUROC) is good at identifying *which* species
is present, but it can't detect the 28 zero-focal-data species (sonotypes +
3 amphibians). A call/no-call gate using acoustic indices would:

1. Filter silent windows before species classification (avoid wasting ensemble
   on quiet periods)
2. Detect biological activity from *any* species, including sonotypes
3. Provide a calibrated activity signal with no domain shift (indices are
   computed directly on soundscape audio)

## Approach

Use the 4 acoustic indices already computed by exp 010:

| Index | What it measures | Expected call/no-call signal |
|-------|-----------------|------------------------------|
| NDSI | Biophony (2-11 kHz) vs anthropophony (1-2 kHz) | Higher = more biological sound |
| Spectral contrast | Peak-to-valley ratio in spectrum | Higher = tonal calls present |
| Spectral flatness | Tonality (0=tonal, 1=noise) | Lower = calls present |
| Bioacoustic index | Energy in 2-15 kHz band | Higher = more vocalizations |

### Analysis plan

**Step 1: Raw index evaluation**
- Load indices for the 66 labeled soundscape files (already in exp010-2026 parquets)
- Aggregate to 5s windows (mean or max of sub-window values)
- Compute AUROC for each index as a standalone call/no-call predictor
- Compare against model-based call scores (ensemble top-5 sum = 0.901 AUROC)

**Step 2: Combined model**
- Logistic regression on 4 indices → call/no-call (739 windows, leave-one-site-out CV)
- Compare against single best index
- Check calibration (reliability diagram)

**Step 3: Gating evaluation**
- Use index-based gate to filter windows, then evaluate species ensemble on
  passing windows only
- Does gating improve precision without hurting recall?

## Ground truth definition

- **Call**: any species annotated in window (all 75 species, including sonotypes)
- **No-call**: window in a labeled file with no species annotated
- 739 labeled windows: 739 call (all have at least one species) + 53 gaps (unlabeled)

Note: with the "any species" definition, all 739 labeled windows are calls.
The 53 unlabeled gaps within labeled files are the only candidate negatives,
plus we could mine hard negatives from the model predictions on unlabeled files.

## Results (2026-03-23)

### Individual indices (792 windows: 739 call, 53 no-call)

| Index | AUROC |
|-------|-------|
| **Spectral contrast (mean)** | **0.888** |
| Bioacoustic index (mean) | 0.850 |
| NDSI (mean) | 0.798 |
| Spectral flatness (std) | 0.793 |

### Combined classifiers all failed

| Classifier | S22 AUROC |
|-----------|-----------|
| GBDT | 0.526 |
| Decision tree d5 | 0.500 |
| Decision tree d3 | 0.489 |
| Logistic regression | 0.361 |
| Ridge | 0.352 |
| Isotonic(spectral contrast) | 0.472 |

All combined models and calibration attempts perform **worse than raw spectral
contrast alone**. Root cause: only 53 negative windows across 2 sites. 6 of 8
CV folds have zero negatives, so classifiers learn "always predict call."

Isotonic regression on spectral contrast also failed — it maps everything to
~0.985 probability, collapsing the ranking that raw values preserve.

### Recommendation

**Use raw spectral contrast with a simple threshold.** No ML, no calibration.
Choose threshold conservatively (e.g., below the 5th percentile of spectral
contrast in labeled call windows).

For pseudo-labeling pipeline: spectral contrast gates "is anything here,"
then the species ensemble identifies what. This covers all species including
the 28 with zero focal data.

## Limitation

Only 53 negative windows (gaps in labeled files). Not enough to train or
calibrate any model. More negatives could be mined from unlabeled soundscapes
(windows where all models agree nothing is present), but that creates a
circular dependency with the species models.
