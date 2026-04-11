# Findings: v1 Pseudo-Label Generation

**Date**: 2026-04-11
**Config**: `configs/v1.yaml`
**Output**: `shared/pseudo-labels/birdclef-2026/v1/`
**Script**: `generate_pseudo_labels.py`

---

## Configuration

| Parameter | Value |
|-----------|-------|
| SED fold strategy | max-across-4-folds |
| SED temperature | 0.6 |
| SED power transform | P^1.5 |
| Perch strategy | per-species z-normalization |
| Perch z-threshold | 2.0 (top ~2.3% per species) |
| Perch min_std | 0.01 |
| Perch power transform | P^1.5 |

---

## Tier Assignment

| Tier | Description | N Species |
|------|-------------|-----------|
| A | Seen in labeled soundscapes | 75 |
| B | Unseen, SED max-folds >0.5 | 109 |
| C | Unseen, SED-blind, Perch signal | 44 (43 active, 1 skipped) |
| D | Invisible to both models | 6 |

**Tier C skipped species**: 1595929 (Uruguay Harlequin Frog) — Perch std=0.0097 < 0.01
threshold, meaning Perch gives essentially identical predictions for all segments (no
signal variation to normalize).

---

## SED Pseudo-Labels (Tiers A+B)

| Metric | Value |
|--------|-------|
| Pseudo-labels >0.5 (unlabeled) | 198,161 |
| Species with coverage >0.5 | 162 / 184 |
| Species with zero coverage >0.5 | 22 |
| Segments with ≥1 SED label >0.5 | 84,002 (66%) |
| Mean SED species per labeled segment | 2.40 |

**22 Tier A+B species lost after P^1.5**: These had SED max-folds max prob between 0.5
and ~0.63. Temperature scaling (T=0.6) pushed them slightly higher, but P^1.5 then
crushed them below 0.5. They still have soft labels in the 0.3-0.5 range — they
contribute minimal gradient but aren't completely lost.

**If we want to recover them**: Lower power transform to P^1.25 in a future variant.
For round 1, these were borderline-confidence species anyway.

---

## Perch Pseudo-Labels (Tier C)

| Metric | Value |
|--------|-------|
| Active species (not skipped) | 43 / 44 |
| Pseudo-labels >0.5 (unlabeled) | 148,707 |
| Species with coverage >0.5 | 34 / 43 |
| Segments per species: min | 0 |
| Segments per species: median | 3,728 |
| Segments per species: max | 8,419 |

### Z-normalization quality check

The critical question: is z-normalization capturing species-specific signal, or shared
Perch noise?

**Pairwise segment overlap between Tier C species (top 15 species checked)**:

| Metric | Value |
|--------|-------|
| Mean overlap | **0.051** |
| Max overlap | 0.196 |
| Min overlap | 0.009 |
| Pairs checked | 105 |

**Mean overlap of 5.1% confirms z-normalization is working.** If different species' top
segments were the same (shared noise), we'd see overlap >50%. Instead, species A's peak
segments are 95% different from species B's — strong evidence of species-specific signal.

### Segment distribution

| Metric | Value |
|--------|-------|
| Segments with 0 Tier C labels >0.5 | 42,038 (33%) |
| Segments with 1+ Tier C labels | 85,858 (67%) |
| Segments with 3+ Tier C labels | 15,560 (12%) |
| Segments with 5+ Tier C labels | 1,778 (1.4%) |
| Max Tier C labels on one segment | 11 |
| Mean Tier C species per segment (when >0) | 1.75 |

Distribution is healthy — no runaway concentration. Most segments get 0-2 Tier C species.

---

## Merged Results

| Metric | Value |
|--------|-------|
| Total pseudo-labels >0.5 (unlabeled) | **346,868** |
| Total pseudo-labels >0.3 (unlabeled) | 382,530 |
| Species with any >0.5 | **196 / 234** |
| Mean labels per segment >0.5 | **2.73** |

### Breakdown by source

| Source | Pseudo-labels >0.5 | Per segment (when >0) |
|--------|--------------------|-----------------------|
| SED (Tiers A+B) | 198,161 | 2.40 |
| Perch (Tier C) | 148,707 | 1.75 |
| **Merged** | **346,868** | **2.73** |

### Context: is 2.73 species/segment reasonable?

- Labeled soundscapes average ~4.2 species per segment (for 75 seen species)
- SED raw predictions: 0.04 unseen species per segment (before pseudo-labeling)
- Perch raw (before z-norm): 119 species per segment (miscalibrated)
- **After pipeline: 2.73** — between the focal/labeled reality and the Perch extreme

This is the right order of magnitude for Pantanal soundscapes (nocturnal, multi-species).

---

## Training Data Balance

| Source | Segments | Loss Weight | Effective Signal |
|--------|----------|-------------|-----------------|
| Focal (train_audio) | 265,924 | 1.0 | 265,924 (80.4%) |
| Labeled soundscapes | 792 | 1.0 | 792 (0.2%) |
| Pseudo-labeled soundscapes | 127,896 | 0.5 | 63,948 (19.3%) |

Pseudo-labels contribute **19.3% of effective training signal** — conservative enough
that errors can't overwhelm ground truth, but substantial enough to matter.

---

## Output Files

| File | Size | Content |
|------|------|---------|
| `pseudo_labels.parquet` | 137 MB | Merged 127K × 236 |
| `sed_pseudo_labels.parquet` | 136 MB | SED component |
| `perch_pseudo_labels.parquet` | 1.6 MB | Perch component (sparse) |
| `tier_assignment.csv` | 2.3 KB | Per-species tier |
| `stats.json` | 12 KB | Full statistics |
| `config.yaml` | 2.0 KB | Copy of v1.yaml |

---

## Assessment

**Ready for Phase B.** The pseudo-labels look well-calibrated:
- SED provides high-precision labels for 162 species (the reliable backbone)
- Perch z-normalization extracts species-specific signal (5% overlap, not shared noise)
- 2.73 species/segment is realistic for multi-species soundscapes
- Training balance is conservative (19% PL signal)
- 196/234 species covered — up from 75 with labeled data only

**Watch points for Phase B**:
- 22 Tier A+B species with borderline coverage — may need lower P^α in round 2
- 9 Tier C species without >0.5 coverage — z-norm helped but not enough for them
- 1 Tier C species skipped entirely (1595929)
- The 6 Tier D species remain at zero — hard floor, accept the loss
