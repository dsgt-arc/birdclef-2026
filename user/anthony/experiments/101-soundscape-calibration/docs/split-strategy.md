# Split Strategy: Site-Level Partitioning

Splits must be **by site**. Windows within the same site share acoustic conditions,
species communities, and equipment.

## Site inventory

| Site | Files | ~Windows | Species | Unlabeled files |
|------|-------|----------|---------|-----------------|
| S22 | 40 | ~480 | 29 | 3,383 |
| S09 | 5 | ~60 | 5 | 12 |
| S08 | 5 | ~60 | 19 | 5 |
| S15 | 4 | ~48 | 14 | 43 |
| S19 | 3 | ~36 | 17 | 76 |
| S23 | 3 | ~36 | 14 | 3 |
| S13 | 2 | ~24 | 6 | 1,873 |
| S18 | 2 | ~24 | 4 | 54 |
| S03 | 2 | ~24 | 4 | 5 |

Window counts approximate (12/file minus unlabeled gaps). Total: 66 files, 739 labeled windows.

## Split

| Split | Sites | Files | ~Windows | Purpose |
|-------|-------|-------|----------|---------|
| **Test** | S22 | 40 | ~480 | Final evaluation only |
| **Train/Val** | remaining 8 | 26 | ~259 | Leave-one-site-out CV |

**Why S22 as test**: largest labeled site (reliable eval), most species (29), largest
unlabeled site (3,383 files → calibration on S22 predicts pseudo-labeling quality).

## Platt scaling granularity

With ~200 train windows per CV fold, most species will have <5 positive windows.
**Per-taxa is the default**, not the fallback.

| Granularity | When | Params |
|-------------|------|--------|
| **Per-taxa** (default) | Always | 2 × 4 evaluable taxa = 8 |
| **Per-species** | Species with ≥5 positives in train fold | 2 per species |
| **Global** | Sanity check | 2 total |

4 evaluable taxa: Aves (28), Amphibia (14), Mammalia (4), Reptilia (1). Insecta
excluded (0 evaluable species). Reptilia and Mammalia will have thin fits — fold
into global if unstable.

## Species coverage gap

Some species may appear only in S22 and not in CV sites. These cannot be calibrated
and fall back to global Platt parameters. Quantify in Phase 1 (task 1.7).
